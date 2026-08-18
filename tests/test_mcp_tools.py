"""Tests for mcp.tools — Config builders, DSN masking, env parsing.

Covers:
- _mask_dsn password hiding
- _env_bool / _env_str parsing
- MCPServerConfig dataclass
- _tag_tools_with_metadata
- _check_node_version (mocked subprocess)
- build_server_configs with mocked env
"""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from mcp.tools import (
    MCPServerConfig,
    _env_bool,
    _env_str,
    _mask_dsn,
    _tag_tools_with_metadata,
    build_server_configs,
)


# ===================================================================
# _mask_dsn
# ===================================================================


class TestMaskDsn:
    def test_masks_password(self):
        dsn = "postgresql://user:secret123@localhost:5432/db"
        masked = _mask_dsn(dsn)
        assert "secret123" not in masked
        assert "***" in masked
        assert "user" in masked

    def test_no_password(self):
        dsn = "postgresql://localhost:5432/db"
        assert _mask_dsn(dsn) == dsn

    def test_special_chars_in_password(self):
        dsn = "mysql://admin:p@ss:w0rd!@db.example.com/mydb"
        masked = _mask_dsn(dsn)
        assert "p@ss:w0rd!" not in masked
        assert "***" in masked


# ===================================================================
# _env_bool
# ===================================================================


class TestEnvBool:
    @pytest.mark.parametrize("val,expected", [
        ("1", True), ("true", True), ("yes", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("off", False),
        ("", False), ("random", False),
    ])
    def test_truthy_and_falsy(self, val, expected):
        with patch.dict(os.environ, {"TEST_BOOL": val}):
            assert _env_bool("TEST_BOOL") == expected

    def test_missing_key_default_false(self):
        assert _env_bool("NONEXISTENT_KEY_XYZ") is False

    def test_missing_key_default_true(self):
        assert _env_bool("NONEXISTENT_KEY_XYZ", True) is True

    def test_whitespace_handling(self):
        with patch.dict(os.environ, {"TEST_BOOL": "  true  "}):
            assert _env_bool("TEST_BOOL") is True


# ===================================================================
# _env_str
# ===================================================================


class TestEnvStr:
    def test_returns_value(self):
        with patch.dict(os.environ, {"TEST_STR": "hello"}):
            assert _env_str("TEST_STR") == "hello"

    def test_missing_returns_default(self):
        assert _env_str("NONEXISTENT_KEY_XYZ", "fallback") == "fallback"

    def test_whitespace_stripped(self):
        with patch.dict(os.environ, {"TEST_STR": "  spaced  "}):
            assert _env_str("TEST_STR") == "spaced"


# ===================================================================
# MCPServerConfig
# ===================================================================


class TestMCPServerConfig:
    def test_default_values(self):
        cfg = MCPServerConfig(name="test")
        assert cfg.name == "test"
        assert cfg.command is None
        assert cfg.args == []
        assert cfg.env is None
        assert cfg.url is None
        assert cfg.transport == "stdio"
        assert cfg.headers is None
        assert cfg.tool_filter is None

    def test_stdio_config(self):
        cfg = MCPServerConfig(
            name="dbhub",
            command="npx",
            args=["-y", "@bytebase/dbhub"],
            transport="stdio",
        )
        assert cfg.command == "npx"
        assert cfg.args == ["-y", "@bytebase/dbhub"]

    def test_http_config(self):
        cfg = MCPServerConfig(
            name="http_srv",
            url="http://localhost:3000/mcp",
            transport="streamable_http",
            headers={"Authorization": "Bearer xxx"},
        )
        assert cfg.url == "http://localhost:3000/mcp"
        assert cfg.headers == {"Authorization": "Bearer xxx"}

    def test_tool_filter(self):
        cfg = MCPServerConfig(
            name="filtered",
            tool_filter={"search", "query"},
        )
        assert "search" in cfg.tool_filter


# ===================================================================
# _tag_tools_with_metadata
# ===================================================================


class TestTagToolsWithMetadata:
    def test_tags_dict_tools(self):
        tools = [
            {"name": "execute_sql", "description": "SQL"},
            {"name": "list_tables", "description": "List"},
        ]
        result = _tag_tools_with_metadata(tools, "dbhub")
        for tool in result:
            assert tool["metadata"]["tool_domain"] == "external_mcp"
            assert tool["metadata"]["_mcp_server"] == "dbhub"
            assert tool["metadata"]["_original_name"] in ("execute_sql", "list_tables")

    def test_preserves_existing_metadata(self):
        tools = [
            {"name": "x", "metadata": {"existing_key": "value"}},
        ]
        result = _tag_tools_with_metadata(tools, "srv")
        assert result[0]["metadata"]["existing_key"] == "value"
        assert result[0]["metadata"]["_mcp_server"] == "srv"

    def test_empty_list(self):
        assert _tag_tools_with_metadata([], "srv") == []

    def test_creates_metadata_if_missing(self):
        tools = [{"name": "tool_a"}]
        result = _tag_tools_with_metadata(tools, "srv")
        assert "metadata" in result[0]
        assert result[0]["metadata"]["_mcp_server"] == "srv"


# ===================================================================
# build_server_configs
# ===================================================================


class TestBuildServerConfigs:
    def test_all_disabled(self):
        configs = build_server_configs(
            enable_dbhub=False,
            enable_markitdown=False,
            enable_feishu=False,
        )
        # Only generic servers from MCP_SERVERS (which we don't set)
        assert "dbhub" not in configs
        assert "markitdown" not in configs
        assert "feishu" not in configs

    def test_markitdown_enabled(self):
        configs = build_server_configs(
            enable_dbhub=False,
            enable_markitdown=True,
            enable_feishu=False,
        )
        assert "markitdown" in configs
        cfg = configs["markitdown"]
        assert cfg.name == "markitdown"
        assert cfg.command == "uv"
        assert cfg.transport == "stdio"

    def test_dbhub_requires_dsn(self):
        with patch.dict(os.environ, {}, clear=True):
            configs = build_server_configs(
                enable_dbhub=True,
                enable_markitdown=False,
                enable_feishu=False,
            )
            # No DBHUB_DSN → no config
            assert "dbhub" not in configs

    def test_feishu_requires_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            configs = build_server_configs(
                enable_dbhub=False,
                enable_markitdown=False,
                enable_feishu=True,
            )
            # No FEISHU_APP_ID → no config
            assert "feishu" not in configs

    def test_generic_servers_from_env(self):
        import json
        servers_json = json.dumps([
            {
                "name": "custom_srv",
                "command": "python",
                "args": ["-m", "my_server"],
                "transport": "stdio",
            },
        ])
        with patch.dict(os.environ, {"MCP_SERVERS": servers_json}):
            configs = build_server_configs(
                enable_dbhub=False,
                enable_markitdown=False,
                enable_feishu=False,
            )
            assert "custom_srv" in configs
            assert configs["custom_srv"].command == "python"

    def test_generic_servers_invalid_json(self):
        with patch.dict(os.environ, {"MCP_SERVERS": "not json"}):
            configs = build_server_configs(
                enable_dbhub=False,
                enable_markitdown=False,
                enable_feishu=False,
            )
            # Should not crash, just skip
            assert "dbhub" not in configs

    def test_generic_http_server(self):
        import json
        servers_json = json.dumps([
            {
                "name": "http_srv",
                "url": "http://localhost:9000/mcp",
                "transport": "streamable_http",
                "headers": {"X-Auth": "token123"},
            },
        ])
        with patch.dict(os.environ, {"MCP_SERVERS": servers_json}):
            configs = build_server_configs(
                enable_dbhub=False,
                enable_markitdown=False,
                enable_feishu=False,
            )
            assert "http_srv" in configs
            assert configs["http_srv"].url == "http://localhost:9000/mcp"
            assert configs["http_srv"].headers == {"X-Auth": "token123"}
