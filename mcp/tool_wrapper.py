"""Tool name conflict resolution via namespace-prefix wrapping.

Multiple MCP servers may expose tools with identical names (e.g., both
DBHub and another server could have a ``search`` tool). This module
provides explicit conflict detection and a safe wrapping strategy that
preserves the original tool's behaviour while giving each tool a unique,
server-scoped name such as ``dbhub__execute_sql``.

Adapted from open_deep_research — works with plain dict tools and MCP SDK
Tool objects instead of LangChain BaseTool.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Characters that are safe in tool names AND MCP tool names.
# We use double-underscore as the separator: dbhub__execute_sql
_SERVER_SEPARATOR = "__"


def prefixed_tool_name(server_name: str, tool_name: str) -> str:
    """Generate a server-scoped tool name.

    Example:
        >>> prefixed_tool_name("dbhub", "execute_sql")
        'dbhub__execute_sql'
    """
    return f"{server_name}{_SERVER_SEPARATOR}{tool_name}"


def _tool_name(tool) -> str:
    """Extract the name from a tool — handles dicts and objects."""
    if isinstance(tool, dict):
        return tool.get("name", "")
    if hasattr(tool, "name"):
        return str(tool.name)
    return ""


def _set_tool_name(tool, name: str) -> None:
    """Set the name on a tool — handles dicts and objects."""
    if isinstance(tool, dict):
        tool["name"] = name
    elif hasattr(tool, "name"):
        # Try model_copy first (Pydantic v2), then direct assignment
        if hasattr(tool, "model_copy"):
            raise TypeError("Immutable tool — use _copy_tool_with_name")
        tool.name = name


def _tool_description(tool) -> str:
    """Extract the description from a tool."""
    if isinstance(tool, dict):
        return tool.get("description", "")
    return getattr(tool, "description", "") or ""


def _copy_tool_with_name(tool, *, name: str, description: str):
    """Create a safe copy of *tool* with a different *name* and *description*.

    Preserves the original coroutine / func, args schema, metadata,
    and other attributes.  No internal fields of the original tool are
    mutated.

    Works with:
    - dict-based tool representations
    - Pydantic v2 objects (model_copy)
    - MCP SDK Tool objects
    - Any object type
    """
    # --- Dict tools ---
    if isinstance(tool, dict):
        copied = dict(tool)
        copied["name"] = name
        copied["description"] = description
        return copied

    # --- Pydantic v2 ---
    if hasattr(tool, "model_copy"):
        return tool.model_copy(update={"name": name, "description": description})

    # --- MCP SDK Tool type ---
    if hasattr(tool, "__class__") and tool.__class__.__name__ == "Tool":
        # Reconstruct via the constructor if available
        if hasattr(tool.__class__, "__init__"):
            import inspect
            sig = inspect.signature(tool.__class__.__init__)
            params = {}
            for param_name in ("name", "description", "inputSchema"):
                if param_name in sig.parameters and hasattr(tool, param_name):
                    params[param_name] = getattr(tool, param_name)
            if "name" in params:
                params["name"] = name
            if "description" in params:
                params["description"] = description
            try:
                return tool.__class__(**params)
            except Exception:
                pass

    # --- Generic object fallback: shallow copy + attribute override ---
    import copy
    copied = copy.copy(tool)
    try:
        copied.name = name
        copied.description = description
    except AttributeError:
        pass
    return copied


def wrap_mcp_tools(
    server_name: str,
    tools: list,
    existing_names: set[str],
) -> list:
    """Namespace *tools* from *server_name* to avoid name collisions.

    Every tool whose original name already appears in *existing_names* is
    renamed to ``{server_name}__{original_name}`` and its description is
    tagged with the server name.  Tools whose names do **not** conflict
    keep their original name — this keeps the prompt short when there are
    no collisions.

    Parameters
    ----------
    server_name:
        Short identifier for the MCP server (e.g. ``"dbhub"``).
    tools:
        MCP tools (dicts or objects) returned by the MCP client.
    existing_names:
        Set of tool names already present in the agent's tool collection.

    Returns
    -------
    list
        The tools, potentially renamed.  **No tools are silently dropped.**
    """
    wrapped: list = []
    for tool in tools:
        original_name = _tool_name(tool)
        if original_name in existing_names:
            prefixed = prefixed_tool_name(server_name, original_name)
            tagged_desc = f"[{server_name}] {_tool_description(tool)}"
            logger.info(
                "MCP tool name conflict: '%s' from server '%s' "
                "→ renaming to '%s'.",
                original_name,
                server_name,
                prefixed,
            )
            renamed = _copy_tool_with_name(
                tool, name=prefixed, description=tagged_desc
            )
            wrapped.append(renamed)
        else:
            wrapped.append(tool)

    return wrapped
