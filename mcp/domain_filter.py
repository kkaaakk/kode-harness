"""Scene-based tool domain filtering and role-based tool routing.

All domain metadata lives in :data:`DOMAIN_REGISTRY` — the single source of
truth.  Keyword lists, prompt labels, and classifier descriptions are all
derived from it, so adding a new domain only requires one update.

Adapted for learn-claude-code — domain definitions match the coding agent's
actual tool landscape (file ops, knowledge, task mgmt, team, memory, MCP).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolLike(Protocol):
    """Minimal tool protocol — anything with a ``name`` and optional ``metadata``."""

    name: str
    metadata: dict[str, Any] | None

# ---------------------------------------------------------------------------
# Domain Registry — single source of truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainDef:
    """Definition of one tool domain.

    Attributes
    ----------
    name:
        Unique key used in metadata and code (e.g. ``"database"``).
    label:
        Human-readable heading for system prompts
        (e.g. ``"Database Tools (DBHub)"``).
    description:
        One-line summary for LLM domain classification in
        ``write_research_brief``.
    always_active:
        If ``True``, this domain is never filtered out (always included).
    keywords:
        Trigger words for keyword-based fallback detection.  Case-insensitive
        match against the research topic.
    """

    name: str
    label: str
    description: str
    always_active: bool = False
    keywords: list[str] = field(default_factory=list)


# fmt: off
DOMAIN_REGISTRY: list[DomainDef] = [
    # ── always active ──────────────────────────────────────────────
    DomainDef(
        name="core",
        label="Core Tools (always available)",
        description="System tools (compress, idle) — always active.",
        always_active=True,
    ),
    DomainDef(
        name="file",
        label="File & Shell",
        description="Read, write, edit files; run bash commands — the coding agent's hands.",
        always_active=True,
    ),
    DomainDef(
        name="planning",
        label="Planning & Tasks",
        description="Todo lists, persistent file tasks with status/owner/dependencies.",
        always_active=True,
    ),
    # ── on-demand ──────────────────────────────────────────────────
    DomainDef(
        name="knowledge",
        label="Knowledge & Skills",
        description="Search codebase via grep/glob, discover and load specialized skill instructions.",
        keywords=[
            "search", "find", "look up", "skill", "knowledge", "grep", "glob",
            "codebase", "how to", "documentation", "reference", "spec",
            "read", "browse", "explore", "locate", "regex", "pattern",
        ],
    ),
    DomainDef(
        name="delegation",
        label="Delegation & Background",
        description="Spawn subagents for exploration/analysis, run background shell tasks.",
        keywords=[
            "subagent", "delegate", "parallel", "background", "explore",
            "research", "investigate", "analyze", "audit", "scan",
            "spawn", "fan out", "concurrent",
        ],
    ),
    DomainDef(
        name="team",
        label="Team Coordination",
        description="Multi-agent team: spawn persistent teammates, message, broadcast, plan approval.",
        keywords=[
            "team", "teammate", "collaborate", "coordinate", "multi-agent",
            "review", "approve", "plan", "orchestrate", "work together",
        ],
    ),
    DomainDef(
        name="memory",
        label="Persistent Memory",
        description="Write, recall, and delete persistent file-based memories across sessions.",
        keywords=[
            "remember", "memory", "preference", "save", "persist",
            "for future", "keep", "store", "note to self",
        ],
    ),
    DomainDef(
        name="external_mcp",
        label="External MCP Tools",
        description="Tools from external MCP servers (DBHub, MarkItDown, Feishu, or custom). Available only when configured.",
        keywords=[
            "database", "sql", "query", "db", "convert", "markdown",
            "feishu", "lark", "document", "pdf",
        ],
    ),
]
# fmt: on

# Fast lookups derived from the registry (kept in sync automatically)
_DOMAIN_BY_NAME: dict[str, DomainDef] = {d.name: d for d in DOMAIN_REGISTRY}
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    d.name: d.keywords for d in DOMAIN_REGISTRY if d.keywords
}
_ALWAYS_ACTIVE_DOMAINS: set[str] = {
    d.name for d in DOMAIN_REGISTRY if d.always_active
}
_DOMAIN_PROMPT_ORDER: list[tuple[str, str]] = [
    (d.name, d.label) for d in DOMAIN_REGISTRY
]

# Built-in (non-MCP) tool name → domain mapping (matched against harness_core.py tools)
_BUILTIN_NAME_TO_DOMAIN: dict[str, str] = {
    # file
    "bash": "file",
    "read_file": "file",
    "write_file": "file",
    "edit_file": "file",
    # knowledge
    "grep_search": "knowledge",
    "glob_search": "knowledge",
    "load_skill": "knowledge",
    # planning
    "TodoWrite": "planning",
    "task_create": "planning",
    "task_get": "planning",
    "task_update": "planning",
    "task_list": "planning",
    "claim_task": "planning",
    # delegation
    "task": "delegation",
    "background_run": "delegation",
    "check_background": "delegation",
    # team
    "spawn_teammate": "team",
    "list_teammates": "team",
    "send_message": "team",
    "read_inbox": "team",
    "broadcast": "team",
    "shutdown_request": "team",
    "plan_approval": "team",
    # memory
    "memory_write": "memory",
    "memory_recall": "memory",
    "memory_delete": "memory",
    # system
    "compress": "core",
    "idle": "core",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_domain(name: str) -> DomainDef | None:
    """Return the :class:`DomainDef` for *name*, or ``None``."""
    return _DOMAIN_BY_NAME.get(name)


def get_domain_description(name: str) -> str:
    """Return the one-line description for *name*."""
    dom = _DOMAIN_BY_NAME.get(name)
    return dom.description if dom else ""


def get_domain_label(name: str) -> str:
    """Return the human-readable label for *name*."""
    dom = _DOMAIN_BY_NAME.get(name)
    return dom.label if dom else name


def iter_domain_labels(domains: set[str]) -> list[tuple[str, str]]:
    """Return ordered ``(name, label)`` pairs for the given *domains*."""
    seen = {name for name, _ in _DOMAIN_PROMPT_ORDER if name in domains}
    # Also include any domains not in the order list
    for name in sorted(domains - seen):
        seen.add(name)
        _DOMAIN_PROMPT_ORDER.append((name, get_domain_label(name)))
    return [(name, label) for name, label in _DOMAIN_PROMPT_ORDER if name in domains]


def build_domain_classifier_prompt() -> str:
    """Generate the domain-classification section for the LLM prompt.

    Used by ``write_research_brief`` so the LLM can output ``relevant_domains``.
    Dynamically built from the registry — no hardcoded domain lists.
    """
    lines: list[str] = []
    for dom in DOMAIN_REGISTRY:
        if dom.always_active:
            continue  # LLM doesn't need to classify always-active domains
        kw_hint = ""
        if dom.keywords:
            kw_hint = f" (keywords: {', '.join(dom.keywords[:6])})"
        lines.append(f"- **{dom.name}**: {dom.description}{kw_hint}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool domain resolution
# ---------------------------------------------------------------------------


def _tool_name(tool) -> str:
    """Extract the name from a tool — handles dicts, objects, and MCP Tool types."""
    if isinstance(tool, dict):
        return tool.get("name", "")
    if hasattr(tool, "name"):
        return str(tool.name)
    return ""


def _tool_metadata(tool) -> dict[str, Any]:
    """Extract metadata from a tool — handles dicts and objects."""
    if isinstance(tool, dict):
        return tool.get("metadata") or {}
    if hasattr(tool, "metadata") and tool.metadata is not None:
        return dict(tool.metadata)
    return {}


def _set_tool_metadata(tool, metadata: dict[str, Any]) -> None:
    """Set metadata on a tool — handles dicts and objects."""
    if isinstance(tool, dict):
        tool["metadata"] = metadata
    elif hasattr(tool, "metadata"):
        tool.metadata = metadata


def _tool_domain(tool) -> str:
    """Resolve the domain of a single tool (dict, MCP tool, or any object with name/metadata).

    Priority:
    1. ``tool.metadata["tool_domain"]`` — set by MCP loader
    2. ``tool["name"]`` / ``tool.name`` → ``_BUILTIN_NAME_TO_DOMAIN``
    3. ``"external_mcp"`` — fallback for anonymous dict / unknown tools
    """
    # 1) MCP-loaded tools have explicit domain metadata
    metadata = _tool_metadata(tool)
    if isinstance(metadata, dict) and "tool_domain" in metadata:
        return str(metadata["tool_domain"])

    # 2) Name-based matching — works for both object and dict tools
    name = _tool_name(tool)
    if name in _BUILTIN_NAME_TO_DOMAIN:
        return _BUILTIN_NAME_TO_DOMAIN[name]

    # 3) Anonymous dict tools → assume external_mcp
    if isinstance(tool, dict):
        return "external_mcp"

    # 4) Fallback
    return "external_mcp"


def tag_builtin_tools(tools: list) -> list:
    """Ensure every built-in tool carries ``tool_domain`` metadata.

    Call this once after assembling the full tool list.
    Tools that already have domain metadata (MCP tools) are left untouched.
    """
    for tool in tools:
        metadata = _tool_metadata(tool)
        if isinstance(metadata, dict) and "tool_domain" not in metadata:
            name = _tool_name(tool)
            domain = _BUILTIN_NAME_TO_DOMAIN.get(name, "external_mcp")
            _set_tool_metadata(tool, {**metadata, "tool_domain": domain})
    return tools


# ---------------------------------------------------------------------------
# Domain detection & filtering
# ---------------------------------------------------------------------------


def classify_tools(tools: list) -> dict[str, list]:
    """Group *tools* by their domain category."""
    buckets: dict[str, list] = {}
    for tool in tools:
        domain = _tool_domain(tool)
        buckets.setdefault(domain, []).append(tool)
    return buckets


def detect_active_domains(
    research_topic: str,
    *,
    agent_role: str = "general_research",
) -> set[str]:
    """Return the set of domains relevant to *research_topic* (keyword fallback)."""
    active = set(_ALWAYS_ACTIVE_DOMAINS)
    text = research_topic.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            active.add(domain)
    logger.debug(
        "Keyword domain detection for role=%s topic=%.120s → %s",
        agent_role, research_topic, sorted(active),
    )
    return active


def filter_tools_by_domain(
    tools: list,
    active_domains: set[str],
) -> list:
    """Return only the tools whose domain is in *active_domains*."""
    filtered: list = []
    dropped: list[str] = []
    for tool in tools:
        domain = _tool_domain(tool)
        if domain in active_domains:
            filtered.append(tool)
        else:
            dropped.append(_tool_name(tool))
    if dropped:
        logger.info(
            "Domain filter: dropped %d tool(s) from inactive domains: %s",
            len(dropped), ", ".join(dropped),
        )
    return filtered


def get_filtered_tools(
    tools: list,
    research_topic: str,
    *,
    agent_role: str = "general_research",
) -> list:
    """Convenience: detect active domains and filter *tools* in one call."""
    active = detect_active_domains(research_topic, agent_role=agent_role)
    return filter_tools_by_domain(tools, active)


def tool_domain_summary(tools: list) -> str:
    """Return a human-readable summary of tools grouped by domain."""
    buckets = classify_tools(tools)
    lines: list[str] = []
    for domain in sorted(buckets):
        names = sorted(_tool_name(t) for t in buckets[domain])
        lines.append(f"  [{domain}] {', '.join(names)}")
    return "\n".join(lines)
