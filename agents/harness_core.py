#!/usr/bin/env python3
"""
harness_core.py - The complete cockpit for the model.

Combines every subsystem into a single agent loop:
  base tools · todos · skills · compression · tasks · subagent ·
  background · team messaging · shutdown / plan protocol

REPL commands: /compact /tasks /team /inbox
"""

import asyncio
import json
import logging
import os
import threading
import uuid
from pathlib import Path
from queue import Queue

# ---------------------------------------------------------------------------
# Re-export config so that ``from agents.harness_core import WORKDIR`` works.
# ---------------------------------------------------------------------------
from agents.config import *  # noqa: F401,F403  (WORKDIR, MODEL, client, ...)
from agents.config import (  # explicit names used in this module
    SKILLS_DIR,
    TOKEN_THRESHOLD,
    VALID_MSG_TYPES,
    WORKDIR,
    client,
    MODEL,
    SANDBOX,
    RUN_MODE,
)

from agents.base_tools import (
    run_bash,
    run_read,
    run_write,
    run_edit,
    run_grep_search,
    run_glob_search,
    BashExecutionResult,
    set_secure_bash_context,
    reset_secure_bash_context,
)
from agents.providers import AnthropicAdapter, ModelRequest, StopReason
# Phase 3A-1: the ONLY model-call path for the Agent Loop. Resolves
# ``client`` lazily from this module's globals so tests that patch
# module.client.messages.create keep working unchanged.
ANTHROPIC_ADAPTER = AnthropicAdapter(client_provider=lambda: client)
from agents.todo_manager import TodoManager
from agents.skill_loader import SkillLoader
from agents.compression import estimate_tokens, microcompact, auto_compact
from agents.task_manager import TaskManager
from agents.subagent import run_subagent
from agents.team_manager import (
    MessageBus,
    TeammateManager,
    shutdown_requests,
    plan_requests,
    handle_shutdown_request,
    handle_plan_review,
)
from agents.extension_system import ExtensionRegistry
from agents.types.events import Event
from agents.tool_registry import (
    ToolRegistry,
    ToolRegistryOverlay,
    LegacyToolRegistryView,
    UnknownToolProfileError,
)
from agents.artifacts import ArtifactStore, ArtifactWriteError
from agents.tool_output_policy import ToolOutputPolicy, OutputPolicyConfig


# === SECTION: background =====================================================
class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self.notifications = Queue()

    def run(self, command: str, timeout: int = 120) -> str:
        tid = str(uuid.uuid4())[:8]
        self.tasks[tid] = {"status": "running", "command": command, "result": None}
        threading.Thread(
            target=self._exec, args=(tid, command, timeout), daemon=True
        ).start()
        return f"Background task {tid} started: {command[:80]}"

    def _exec(self, tid: str, command: str, timeout: int):
        try:
            output = SANDBOX.execute(command, timeout=timeout, cwd=str(WORKDIR))
            output = output[:50000] if output else "(no output)"
            self.tasks[tid].update({"status": "completed", "result": output})
        except Exception as e:
            self.tasks[tid].update({"status": "error", "result": str(e)})
        self.notifications.put(
            {
                "task_id": tid,
                "status": self.tasks[tid]["status"],
                "result": self.tasks[tid]["result"][:500],
            }
        )

    def check(self, tid: str = None) -> str:
        if tid:
            t = self.tasks.get(tid)
            return (
                f"[{t['status']}] {t.get('result') or '(running)'}"
                if t
                else f"Unknown: {tid}"
            )
        return (
            "\n".join(
                f"{k}: [{v['status']}] {v['command'][:60]}"
                for k, v in self.tasks.items()
            )
            or "No bg tasks."
        )

    def drain(self) -> list:
        notifs = []
        while not self.notifications.empty():
            notifs.append(self.notifications.get_nowait())
        return notifs


# === SECTION: global_instances ================================================
TODO = TodoManager()
SKILLS = SkillLoader(SKILLS_DIR)
TASK_MGR = TaskManager()
BG = BackgroundManager()
BUS = MessageBus()
TEAM = TeammateManager(BUS, TASK_MGR)

# Extension registry — stage 1 wiring.
# Default: empty registry. emit() returns a no-op outcome, so agent behavior
# is unchanged unless extensions are registered. Kernel safety (dangerous
# command blocking in run_bash) is NOT moved here; it stays in base_tools.
EXTENSIONS = ExtensionRegistry()

# === SECTION: system_prompt ===================================================
SYSTEM = f"""You are a coding agent at {WORKDIR}. Use tools to solve tasks.
Prefer task_create/task_update/task_list for multi-step work. Use TodoWrite for short checklists.
Use task for subagent delegation. Use load_skill for specialized knowledge.
Skills: {SKILLS.descriptions()}"""


# === SECTION: tool_dispatch ===================================================
# Stage 2A: all tools are registered in TOOL_REGISTRY, then TOOLS and
# TOOL_HANDLERS are DERIVED from the registry. This keeps tool lookup and
# schema generation centralized while preserving backward compatibility:
#   - TOOLS list: same order, same schemas as before
#   - TOOL_HANDLERS dict: same name -> handler mapping
#   - agent_loop still uses TOOLS and TOOL_HANDLERS (no logic change)
# ToolRegistry does NOT enforce security; Kernel safety stays in run_bash.

# Step 1: define (name, description, schema, handler) tuples in the EXACT
# order they appeared in the legacy TOOLS list. Order matters because the
# model sees tools in this order.
# ---------------------------------------------------------------------------
# Stage 2D-B: TodoWrite migrated from Base to TodoExtension.
#
# LEGACY_25_TOOL_NAMES is the canonical pre-2D-B tool order (the stage 2A
# hard contract: "default tool order unchanged"). Base tools are registered
# with ``order = LEGACY_25_TOOL_NAMES.index(name)``, so the 24 Base tools
# occupy slots 0,1,2,3,5,6,...,24 (slot 4 is reserved for TodoWrite).
# TodoExtension registers TodoWrite with ``order=LEGACY_TODO_WRITE_ORDER``
# (== 4) so the default composed registry resolves to EXACTLY
# LEGACY_25_TOOL_NAMES — the model still sees the same 25 tools in the
# same order as before the migration.
# ---------------------------------------------------------------------------

LEGACY_25_TOOL_NAMES = [
    "bash", "read_file", "write_file", "edit_file", "TodoWrite",
    "task", "load_skill", "compress", "background_run", "check_background",
    "task_create", "task_get", "task_update", "task_list",
    "spawn_teammate", "list_teammates", "send_message", "read_inbox",
    "broadcast", "shutdown_request", "plan_approval", "idle", "claim_task",
    "grep_search", "glob_search",
]

# Stage 2D-D3: single source of truth for legacy tool order slots.
# Every extension registers with ``order=LEGACY_TOOL_ORDER[<name>]``; Base
# tools use ``LEGACY_TOOL_ORDER[<name>]`` as well (all 25 slots are pinned
# here, including the 12 Base slots). The per-tool LEGACY_*_ORDER constants
# below are kept as thin aliases for backward compatibility - they read
# from this dict, so there is exactly ONE place that pins the order.
# Values are MECHANICALLY identical to the pre-D3 constants (0-indexed
# slots in LEGACY_25_TOOL_NAMES); D3 changes no order values.
LEGACY_TOOL_ORDER = {name: idx for idx, name in enumerate(LEGACY_25_TOOL_NAMES)}

# The slot TodoWrite occupied in the pre-2D-B _TOOL_DEFS list (0-indexed).
LEGACY_TODO_WRITE_ORDER = LEGACY_TOOL_ORDER["TodoWrite"]  # == 4

# Schema for the TodoWrite tool. Unchanged from pre-2D-B; only the
# registration ownership moved from Base to TodoExtension.
TODO_WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                    },
                    "activeForm": {"type": "string"},
                },
                "required": ["content", "status", "activeForm"],
            },
        }
    },
    "required": ["items"],
}


# ---------------------------------------------------------------------------
# Stage 2D-C: the four task tools (task_create/get/update/list) migrated
# from Base to TaskExtension. Schemas are unchanged from pre-2D-C; only the
# registration ownership moved (same pattern as TodoWrite in 2D-B).
# LEGACY_*_ORDER pins each tool to its original slot in LEGACY_25_TOOL_NAMES
# so the default composed registry (Base + TodoExtension + TaskExtension)
# resolves to EXACTLY the legacy 25-tool order.
# ---------------------------------------------------------------------------
TASK_CREATE_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["subject"],
}

TASK_GET_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "integer"},
    },
    "required": ["task_id"],
}

TASK_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "integer"},
        "status": {
            "type": "string",
            "enum": ["pending", "in_progress", "completed", "deleted"],
        },
        "add_blocked_by": {"type": "array", "items": {"type": "integer"}},
        "remove_blocked_by": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["task_id"],
}

TASK_LIST_SCHEMA = {
    "type": "object",
    "properties": {},
}

# The slots the four task tools occupied in the pre-2D-C _TOOL_DEFS list
# (0-indexed). TaskExtension registers with these orders so the default
# composed registry keeps the legacy 25-tool order.
LEGACY_TASK_CREATE_ORDER = LEGACY_TOOL_ORDER["task_create"]  # == 10
LEGACY_TASK_GET_ORDER = LEGACY_TOOL_ORDER["task_get"]        # == 11
LEGACY_TASK_UPDATE_ORDER = LEGACY_TOOL_ORDER["task_update"]  # == 12
LEGACY_TASK_LIST_ORDER = LEGACY_TOOL_ORDER["task_list"]      # == 13


# ---------------------------------------------------------------------------
# Stage 2D-D1: the ``task`` tool (subagent) migrated from Base to
# SubagentExtension. Schema is unchanged from pre-2D-D1; only registration
# ownership moved (same pattern as TodoWrite/Task tools in 2D-B/2D-C).
# run_subagent()'s internal execution model is NOT changed — it keeps its
# own 30-round loop, hardcoded child tool set, and synchronous same-Task
# execution. This extension is purely a registration move.
# ---------------------------------------------------------------------------
TASK_SUBAGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "agent_type": {
            "type": "string",
            "enum": ["Explore", "general-purpose"],
        },
    },
    "required": ["prompt"],
}

# The slot the "task" (subagent) tool occupied in the pre-2D-D1 _TOOL_DEFS
# list (0-indexed == 5). SubagentExtension registers with this order so the
# default composed registry keeps the legacy 25-tool order.
LEGACY_SUBAGENT_ORDER = LEGACY_TOOL_ORDER["task"]  # == 5


# ---------------------------------------------------------------------------
# Stage 2D-D2: the seven Parent-visible Team tools migrated from Base to
# TeamExtension. Schemas are unchanged from pre-2D-D2; only registration
# ownership moved (same pattern as TodoWrite/Task/Subagent in 2D-B/2D-C/2D-D1).
# LEGACY_*_ORDER pins each tool to its original slot in LEGACY_25_TOOL_NAMES
# so the default composed registry (Base + TodoExtension + TaskExtension +
# SubagentExtension + TeamExtension) resolves to EXACTLY the legacy 25-tool
# order.
#
# IMPORTANT: these are the PARENT Agent's Team management tools — NOT the
# Team member's internal hardcoded tool set. Team member _loop() builds its
# own tools inline (bash/read_file/write_file/edit_file/send_message/idle/
# claim_task) and is NOT affected by this migration. See D0 contract
# snapshot for the full Parent→Child inheritance matrix.
# ---------------------------------------------------------------------------
SPAWN_TEAMMATE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "role": {"type": "string"},
        "prompt": {"type": "string"},
    },
    "required": ["name", "role", "prompt"],
}

LIST_TEAMMATES_SCHEMA = {
    "type": "object",
    "properties": {},
}

SEND_MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "to": {"type": "string"},
        "content": {"type": "string"},
        "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)},
    },
    "required": ["to", "content"],
}

READ_INBOX_SCHEMA = {
    "type": "object",
    "properties": {},
}

BROADCAST_SCHEMA = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
    },
    "required": ["content"],
}

SHUTDOWN_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "teammate": {"type": "string"},
    },
    "required": ["teammate"],
}

PLAN_APPROVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "request_id": {"type": "string"},
        "approve": {"type": "boolean"},
        "feedback": {"type": "string"},
    },
    "required": ["request_id", "approve"],
}

LEGACY_SPAWN_TEAMMATE_ORDER = LEGACY_TOOL_ORDER["spawn_teammate"]      # == 14
LEGACY_LIST_TEAMMATES_ORDER = LEGACY_TOOL_ORDER["list_teammates"]      # == 15
LEGACY_SEND_MESSAGE_ORDER = LEGACY_TOOL_ORDER["send_message"]          # == 16
LEGACY_READ_INBOX_ORDER = LEGACY_TOOL_ORDER["read_inbox"]              # == 17
LEGACY_BROADCAST_ORDER = LEGACY_TOOL_ORDER["broadcast"]                # == 18
LEGACY_SHUTDOWN_REQUEST_ORDER = LEGACY_TOOL_ORDER["shutdown_request"]  # == 19
LEGACY_PLAN_APPROVAL_ORDER = LEGACY_TOOL_ORDER["plan_approval"]        # == 20


_TOOL_DEFS = [
    ("bash", "Run a shell command.",
     {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
     lambda **kw: run_bash(kw["command"]), "shell"),
    ("read_file", "Read file contents.",
     {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]},
     lambda **kw: run_read(kw["path"], kw.get("limit")), "file_read"),
    ("write_file", "Write content to file.",
     {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
     lambda **kw: run_write(kw["path"], kw["content"]), "file_write"),
    ("edit_file", "Replace exact text in file.",
     {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]},
     lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]), "file_write"),
    ("load_skill", "Load specialized knowledge by name.",
     {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
     lambda **kw: SKILLS.load(kw["name"]), "default"),
    ("compress", "Manually compress conversation context.",
     {"type": "object", "properties": {}},
     lambda **kw: "Compressing...", "default"),
    ("background_run", "Run command in background thread.",
     {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]},
     lambda **kw: BG.run(kw["command"], kw.get("timeout", 120)), "shell"),
    ("check_background", "Check background task status.",
     {"type": "object", "properties": {"task_id": {"type": "string"}}},
     lambda **kw: BG.check(kw.get("task_id")), "default"),
    ("idle", "Enter idle state.",
     {"type": "object", "properties": {}},
     lambda **kw: "Lead does not idle.", "default"),
    ("claim_task", "Claim a task from the board.",
     {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]},
     lambda **kw: TASK_MGR.claim(kw["task_id"], "lead"), "default"),
    ("grep_search", "Search file contents for a regex pattern across workspace files. Returns matching lines with file paths and line numbers. Use this to find code, definitions, or usage patterns.",
     {"type": "object", "properties": {"pattern": {"type": "string", "description": "Regex pattern to search for"}, "path": {"type": "string", "description": "Directory to search within (relative to workspace). Default: '.'"}, "include": {"type": "string", "description": "Glob pattern to filter files (e.g. '*.py'). Empty means all files."}, "ignore_case": {"type": "boolean", "description": "Case-insensitive search. Default: false"}, "max_results": {"type": "integer", "description": "Maximum matches to return. Default: 50"}}, "required": ["pattern"]},
     lambda **kw: run_grep_search(kw["pattern"], kw.get("path", "."), kw.get("include", ""), kw.get("ignore_case", False), kw.get("max_results", 50)), "default"),
    ("glob_search", "Find files by name pattern (glob) within the workspace. Returns matching file paths relative to workspace root.",
     {"type": "object", "properties": {"pattern": {"type": "string", "description": "Glob pattern for file names (e.g. '*.py', 'test_*.py', '**/config/*.json')"}, "path": {"type": "string", "description": "Base directory to search from (relative to workspace). Default: '.'"}, "max_results": {"type": "integer", "description": "Maximum paths to return. Default: 100"}}, "required": ["pattern"]},
     lambda **kw: run_glob_search(kw["pattern"], kw.get("path", "."), kw.get("max_results", 100)), "default"),
]

# Step 2: create the global Base ToolRegistry and register all tools.
# Stage 2D-A: renamed TOOL_REGISTRY → BASE_TOOL_REGISTRY to make explicit
# that this is the immutable Base registry. Per-call Extension tools go
# into a ToolRegistryOverlay created at agent_loop() startup.
# Stage 2D-B/2D-C/2D-D1/2D-D2: TodoWrite, the four task tools, the subagent
# ``task`` tool, and the seven Team management tools are NO LONGER in Base.
# Base has 12 tools. Each Base tool is
# registered with ``order = LEGACY_25_TOOL_NAMES.index(name)`` so the
# default composed registry (Base + TodoExtension + TaskExtension +
# SubagentExtension + TeamExtension) resolves to LEGACY_25_TOOL_NAMES
# exactly — preserving the stage 2A order contract.
# overwrite=False (default): duplicate registration raises ValueError.
BASE_TOOL_REGISTRY = ToolRegistry()
for _name, _desc, _schema, _handler, _perm in _TOOL_DEFS:
    BASE_TOOL_REGISTRY.register(
        name=_name,
        description=_desc,
        input_schema=_schema,
        handler=_handler,
        permission=_perm,
        order=LEGACY_TOOL_ORDER[_name],
    )


# ---------------------------------------------------------------------------
# Stage 2D-B: TodoExtension — owns the TodoWrite tool registration.
#
# TodoExtension is stateless: it does NOT hold per-agent todo data. The
# todo store is the module-level ``TODO`` TodoManager instance (unchanged
# from pre-2D-B). Keeping the extension stateless means the default
# singleton in DEFAULT_TOOL_CONTRIBUTORS cannot leak state across sessions.
# ---------------------------------------------------------------------------

class TodoExtension:
    """Contributes the TodoWrite tool to a per-call overlay.

    Stateless by design: the handler closes over the module-level ``TODO``
    TodoManager — the same store used pre-2D-B. This extension only moves
    the tool's *registration ownership* from Base to an extension; it does
    NOT change schema, handler, todo storage, or return format.
    """

    extension_id = "todo-extension"

    def contribute_tools(self, registry) -> None:
        registry.register(
            name="TodoWrite",
            description="Update task tracking list.",
            input_schema=TODO_WRITE_SCHEMA,
            handler=lambda **kw: TODO.update(kw["items"]),
            owner=self.extension_id,
            source="extension",
            order=LEGACY_TODO_WRITE_ORDER,
        )


class TaskExtension:
    """Contributes the four task tools (task_create/get/update/list) to a
    per-call overlay.

    Stateless by design: handlers close over the module-level ``TASK_MGR``
    TaskManager — the same file-backed store used pre-2D-C. Task state is
    Harness-level (a shared task board in ``TASKS_DIR``), NOT per-session;
    this migration does NOT change that semantic. This extension only moves
    tool *registration ownership* from Base to an extension; it does NOT
    change schema, handler, task storage, or return format.
    """

    extension_id = "task-extension"

    def contribute_tools(self, registry) -> None:
        registry.register(
            name="task_create",
            description="Create a persistent file task.",
            input_schema=TASK_CREATE_SCHEMA,
            handler=lambda **kw: TASK_MGR.create(
                kw["subject"], kw.get("description", "")
            ),
            owner=self.extension_id,
            source="extension",
            order=LEGACY_TASK_CREATE_ORDER,
        )
        registry.register(
            name="task_get",
            description="Get task details by ID.",
            input_schema=TASK_GET_SCHEMA,
            handler=lambda **kw: TASK_MGR.get(kw["task_id"]),
            owner=self.extension_id,
            source="extension",
            order=LEGACY_TASK_GET_ORDER,
        )
        registry.register(
            name="task_update",
            description="Update task status or dependencies.",
            input_schema=TASK_UPDATE_SCHEMA,
            handler=lambda **kw: TASK_MGR.update(
                kw["task_id"],
                kw.get("status"),
                kw.get("add_blocked_by"),
                kw.get("remove_blocked_by"),
            ),
            owner=self.extension_id,
            source="extension",
            order=LEGACY_TASK_UPDATE_ORDER,
        )
        registry.register(
            name="task_list",
            description="List all tasks.",
            input_schema=TASK_LIST_SCHEMA,
            handler=lambda **kw: TASK_MGR.list_all(),
            owner=self.extension_id,
            source="extension",
            order=LEGACY_TASK_LIST_ORDER,
        )


class SubagentExtension:
    """Contributes the ``task`` (subagent) tool to a per-call overlay.

    Stateless by design: the handler delegates to ``run_subagent`` — the
    SAME function used pre-2D-D1. This extension ONLY moves registration
    ownership from Base to an extension; it does NOT change run_subagent's
    internal execution model in any way:

      - Still synchronous, running in the Parent agent_loop's asyncio.Task.
      - Still its own 30-round loop (NOT a recursive agent_loop call).
      - Still a hardcoded child tool set (bash/read_file[+write/edit]).
      - Still does NOT use ToolRegistry / tool_profile / tool_contributors.
      - Still shares the global Client / Model / Sandbox.
      - Still reuses the Parent's SecureBashContext (same Task → same
        ContextVar + live nonce); does NOT call set/reset_secure_bash_context.

    Any change to run_subagent's runtime behavior is a separate, later
    decision (e.g. a future "Runtime Unification" phase) — never a
    side-effect of this registration migration.
    """

    extension_id = "subagent-extension"

    def contribute_tools(self, registry) -> None:
        registry.register(
            name="task",
            description="Spawn a subagent for isolated exploration or work.",
            input_schema=TASK_SUBAGENT_SCHEMA,
            handler=lambda **kw: run_subagent(
                kw["prompt"], kw.get("agent_type", "Explore")
            ),
            owner=self.extension_id,
            source="extension",
            order=LEGACY_SUBAGENT_ORDER,
        )


class TeamExtension:
    """Contributes the seven Parent-visible Team management tools to a
    per-call overlay.

    Stateless by design: handlers close over the module-level ``TEAM``
    (TeammateManager), ``BUS`` (MessageBus), and the ``handle_shutdown_request``
    / ``handle_plan_review`` helpers — the same objects used pre-2D-D2.
    Team state (member roster, message bus, shutdown/plan requests) is
    Harness-level, NOT per-session; this migration does NOT change that
    semantic. This extension ONLY moves tool *registration ownership* from
    Base to an extension; it does NOT change schema, handler, team storage,
    or return format.

    CRITICAL: these are the PARENT Agent's Team management tools. The Team
    member's internal _loop() builds its OWN hardcoded tool set inline and
    is completely unaffected by this migration. Team member _loop() does
    NOT use ToolRegistry, does NOT call agent_loop, and does NOT inherit
    Parent Profile / Contributors. See the D0 contract snapshot for the
    full Parent→Child inheritance matrix.
    """

    extension_id = "team-extension"

    def contribute_tools(self, registry) -> None:
        registry.register(
            name="spawn_teammate",
            description="Spawn a persistent autonomous teammate.",
            input_schema=SPAWN_TEAMMATE_SCHEMA,
            handler=lambda **kw: TEAM.spawn(
                kw["name"], kw["role"], kw["prompt"]
            ),
            owner=self.extension_id,
            source="extension",
            order=LEGACY_SPAWN_TEAMMATE_ORDER,
        )
        registry.register(
            name="list_teammates",
            description="List all teammates.",
            input_schema=LIST_TEAMMATES_SCHEMA,
            handler=lambda **kw: TEAM.list_all(),
            owner=self.extension_id,
            source="extension",
            order=LEGACY_LIST_TEAMMATES_ORDER,
        )
        registry.register(
            name="send_message",
            description="Send a message to a teammate.",
            input_schema=SEND_MESSAGE_SCHEMA,
            handler=lambda **kw: BUS.send(
                "lead", kw["to"], kw["content"], kw.get("msg_type", "message")
            ),
            owner=self.extension_id,
            source="extension",
            order=LEGACY_SEND_MESSAGE_ORDER,
        )
        registry.register(
            name="read_inbox",
            description="Read and drain the lead's inbox.",
            input_schema=READ_INBOX_SCHEMA,
            handler=lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
            owner=self.extension_id,
            source="extension",
            order=LEGACY_READ_INBOX_ORDER,
        )
        registry.register(
            name="broadcast",
            description="Send message to all teammates.",
            input_schema=BROADCAST_SCHEMA,
            handler=lambda **kw: BUS.broadcast(
                "lead", kw["content"], TEAM.member_names()
            ),
            owner=self.extension_id,
            source="extension",
            order=LEGACY_BROADCAST_ORDER,
        )
        registry.register(
            name="shutdown_request",
            description="Request a teammate to shut down.",
            input_schema=SHUTDOWN_REQUEST_SCHEMA,
            handler=lambda **kw: handle_shutdown_request(BUS, kw["teammate"]),
            owner=self.extension_id,
            source="extension",
            order=LEGACY_SHUTDOWN_REQUEST_ORDER,
        )
        registry.register(
            name="plan_approval",
            description="Approve or reject a teammate's plan.",
            input_schema=PLAN_APPROVAL_SCHEMA,
            handler=lambda **kw: handle_plan_review(
                BUS, kw["request_id"], kw["approve"], kw.get("feedback", "")
            ),
            owner=self.extension_id,
            source="extension",
            order=LEGACY_PLAN_APPROVAL_ORDER,
        )


# The default set of optional tool contributors. ``agent_loop`` uses this
# when the caller passes ``tool_contributors=None`` (the new default), so
# the default behavior still exposes TodoWrite + the four task tools + the
# subagent ``task`` tool + the seven Team management tools. Passing ``()``
# explicitly disables ALL optional extensions — TodoWrite, the task tools,
# ``task``, and all Team tools are then unknown.
DEFAULT_TOOL_CONTRIBUTORS: tuple = (
    TodoExtension(),
    TaskExtension(),
    SubagentExtension(),
    TeamExtension(),
)


def build_default_tool_registry() -> ToolRegistryOverlay:
    """Build the default composed registry: Base + DEFAULT_TOOL_CONTRIBUTORS.

    Returns a fresh overlay over BASE_TOOL_REGISTRY with the default
    contributors applied. ``TOOLS`` / ``TOOL_HANDLERS`` / ``TOOL_REGISTRY``
    are derived from this so legacy callers still see the full 25-tool set
    in the original order. ``agent_loop`` does NOT reuse this module-level
    overlay — it builds its own per-call overlay so concurrent runs never
    share mutable state.
    """
    overlay = ToolRegistryOverlay(BASE_TOOL_REGISTRY)
    for _contributor in DEFAULT_TOOL_CONTRIBUTORS:
        _contribute = getattr(_contributor, "contribute_tools", None)
        if _contribute is not None:
            _contribute(overlay)
    return overlay


_DEFAULT_COMPOSED_REGISTRY = build_default_tool_registry()

# Backward-compat alias so existing tests / imports keep working.
# Stage 2D-B.1: TOOL_REGISTRY is now a READ-ONLY LegacyToolRegistryView over
# the default composed registry (Base + default extensions), NOT the mutable
# overlay. Legacy callers can iterate tool names, index entries, resolve
# profiles, and query handlers — but register/unregister/clear raise
# TypeError (use BASE_TOOL_REGISTRY for mutation, or build_default_tool_registry()
# for a fresh composed overlay). This fixes the stage 2A debt where
# set(TOOL_REGISTRY) / TOOL_REGISTRY[name] raised TypeError.
TOOL_REGISTRY = LegacyToolRegistryView(_DEFAULT_COMPOSED_REGISTRY)

# Step 3: derive TOOLS and TOOL_HANDLERS from the default COMPOSED registry
# (Base + TodoExtension), so legacy callers still see the full 25-tool set
# in the original order. profile=None returns all visible tools.
TOOLS = _DEFAULT_COMPOSED_REGISTRY.resolve(profile=None)
TOOL_HANDLERS = _DEFAULT_COMPOSED_REGISTRY.resolve_handlers(profile=None)

# Stage 2D-B/2D-C/2D-D1/2D-D2 sanity checks (fail fast at import time if the
# migration contract is broken):
# - Base registry has 12 tools (TodoWrite + 4 task tools + subagent + 7 Team
#   tools all removed).
# - Default composed registry has 25 tools (all contributed back by extensions).
# - Default composed order matches the pre-2D-B legacy order exactly.
assert len(_TOOL_DEFS) == 12, (
    f"Expected 12 Base tool defs after TodoWrite + task + subagent + team migration, "
    f"got {len(_TOOL_DEFS)}"
)
assert len(BASE_TOOL_REGISTRY.resolve(profile=None)) == 12, (
    "BASE_TOOL_REGISTRY must have 12 tools after TodoWrite + task + subagent + team migration"
)
assert len(TOOLS) == 25 and len(TOOL_HANDLERS) == 25, (
    f"TOOLS/TOOL_HANDLERS must keep 25 entries, got {len(TOOLS)}/{len(TOOL_HANDLERS)}"
)
_composed_names = [t["name"] for t in TOOLS]
assert _composed_names == LEGACY_25_TOOL_NAMES, (
    f"Default composed tool order mismatch:\n"
    f"  got:      {_composed_names}\n"
    f"  expected: {LEGACY_25_TOOL_NAMES}"
)


# === SECTION: tool_output_policy ==============================================
# Stage 2C-B1: ArtifactStore + ToolOutputPolicy for large tool outputs.
# Stage 2C-B2B-1: Artifact root is PRIVATE — it lives OUTSIDE WORKDIR so
# that bash (which runs in WORKDIR) cannot reach other sessions' artifact
# files by guessing the filesystem path. The model only sees
# artifact://<id> URIs; physical access is mediated by ArtifactStore.
#
# Location policy:
#   - On Windows: <tempdir>/learn-claude-code-artifacts/<workdir-hash>
#   - On POSIX:   /tmp/learn-claude-code-artifacts/<workdir-hash>  (or $TMPDIR)
# The <workdir-hash> segregates different projects so concurrent agents
# on different workdirs don't collide. Session subdirs under the root
# provide per-session isolation.
import hashlib as _hashlib
import tempfile as _tempfile
import uuid as _uuid


def _resolve_artifact_root(workdir: Path) -> Path:
    """Compute a private artifact root OUTSIDE the given workdir.

    Stage 2C-B2B-4: the root location is chosen as follows:

    1. If ``AGENT_ARTIFACT_ROOT`` env var is set, use that path verbatim
       (caller takes responsibility for placing it outside all Docker
       mount sources). Useful for deployments that want a fixed
       private location.
    2. Otherwise, default to ``<tempdir>/learn-claude-code-artifacts/<workdir-hash>``.
       The <workdir-hash> segregates different projects so concurrent
       agents on different workdirs don't collide.

    The root is stable for a given workdir (so repeated agent_loop calls
    in the same project share the root) but distinct across workdirs.

    Permissions: on POSIX we try to chmod the root to 0700 so other
    OS users cannot read other sessions' artifacts. On Windows the
    tempdir ACL is relied upon (chmod is a no-op). Best-effort: if
    chmod fails we proceed, because the Docker mount-plan check is the
    primary isolation mechanism, not filesystem permissions.
    """
    env_root = os.getenv("AGENT_ARTIFACT_ROOT")
    if env_root:
        base = Path(env_root)
    else:
        wd_hash = _hashlib.sha1(
            str(workdir.resolve()).encode("utf-8")
        ).hexdigest()[:12]
        base = Path(_tempfile.gettempdir()) / "learn-claude-code-artifacts" / wd_hash
    base.mkdir(parents=True, exist_ok=True)
    # Best-effort restrictive permissions on the root directory.
    try:
        os.chmod(base, 0o700)
    except OSError:
        # Windows or permission issue — Docker mount-plan check is the
        # primary isolation mechanism, not filesystem perms. Proceed.
        pass
    return base


_ARTIFACT_ROOT = _resolve_artifact_root(WORKDIR)
_ARTIFACT_SESSION_ID = _uuid.uuid4().hex[:16]
ARTIFACT_STORE = ArtifactStore(
    root_dir=_ARTIFACT_ROOT,
    session_id=_ARTIFACT_SESSION_ID,
)
TOOL_OUTPUT_POLICY = ToolOutputPolicy(
    store=ARTIFACT_STORE,
    config=OutputPolicyConfig(),
    workdir=WORKDIR,
    artifact_root=_ARTIFACT_ROOT,
    session_id=_ARTIFACT_SESSION_ID,
)

# Tools whose outputs are processed by the policy.
# Stage 2C-B1: read_file.
# Stage 2C-B2A: bash (structured BashExecutionResult).
# Stage 2C-B3B: grep_search (structured GrepSearchResult → JSONL artifact).
_OUTPUT_POLICY_TOOLS: set[str] = {"read_file", "bash", "grep_search"}


def _is_tool_error(output) -> bool:
    """Return True if a tool's output represents a transport-level error
    that should bypass OutputPolicy and be recorded as tool_error.

    Stage 2C-B2A: a BashExecutionResult with non-zero exit_code is a
    command-level failure, NOT a transport error — the model should still
    see it through OutputPolicy (with exit_code preserved in metadata).
    Only string outputs starting with "Error:" (dangerous-command block,
    timeout, unknown tool, etc.) count as transport errors here.
    """
    if isinstance(output, str):
        return output.startswith("Error")
    return False


# === SECTION: agent_loop ======================================================
def _invoke_handler(block_name, effective_args, active_handlers, tool_profile,
                    call_registry=None):
    """Shared handler-execution path. Returns the tool output string.

    Stage 2D-A: ``call_registry`` is the per-call overlay (Base + Extension
    tools). When provided, it's used to distinguish "tool exists but not
    in this profile" from "unknown tool". When None (legacy callers),
    falls back to BASE_TOOL_REGISTRY.
    """
    handler = active_handlers.get(block_name)
    if handler is not None:
        try:
            return handler(**effective_args)
        except Exception as e:
            return f"Error: {e}"
    # Tool not in active_handlers. Distinguish "registered but profile
    # excluded it" from "truly unknown".
    registry = call_registry if call_registry is not None else BASE_TOOL_REGISTRY
    if registry.has(block_name):
        return (f"Tool unavailable in current profile "
                f"({tool_profile!r}): {block_name}")
    return f"Unknown tool: {block_name}"


# Tools whose ``path`` argument is resolved against WORKDIR and therefore
# must be guarded against artifact-root access. bash is NOT in this set:
# its ``command`` argument is arbitrary shell and cannot be statically
# guarded — bash artifact isolation is the responsibility of 2C-B2 (via
# a structured BashExecutionResult and a sandbox/path whitelist).
_PATH_BASED_TOOLS: frozenset[str] = frozenset({
    "read_file", "write_file", "edit_file", "grep_search", "glob_search",
})


def _path_resolves_into_artifact_root(path_str: str) -> bool:
    """Return True if ``path_str`` (resolved against WORKDIR) falls inside
    ``_ARTIFACT_ROOT``.

    Unconditional normalization: we do NOT string-match
    ".harness/artifacts/" because backslash, ``..`` traversal, symlinks,
    and absolute-path variants would bypass a string check. We resolve
    and compare with os.path.commonpath.

    Returns False on unresolvable paths (caller will let the normal
    handler produce its own error).
    """
    if not isinstance(path_str, str) or not path_str:
        return False
    try:
        candidate = (WORKDIR / path_str).resolve()
    except (OSError, ValueError):
        return False
    try:
        root_resolved = _ARTIFACT_ROOT.resolve()
    except (OSError, ValueError):
        return False
    try:
        common = Path(os.path.commonpath([str(candidate), str(root_resolved)]))
    except ValueError:
        # Different Windows drives — not inside root.
        return False
    return common == root_resolved


def _execute_tool_or_artifact_read(
    block_name,
    effective_args,
    active_handlers,
    tool_profile,
    *,
    call_store: ArtifactStore,
    call_session_id: str,
    call_registry=None,
):
    """Stage 2C-B1.4 tool dispatch with artifact-root guard.

    Cases:
      - read_file with ``artifact://<id>``: route via the per-call store.
        The store only resolves within ``call_session_id``'s directory,
        so cross-session access is structurally impossible.
      - any tool in ``_PATH_BASED_TOOLS`` whose ``path`` resolves into
        ``_ARTIFACT_ROOT``: denied. The model must use ``artifact://`` URIs
        (for read_file) or stay out of the artifact directory (for
        write/edit/grep/glob). Direct FS paths into ANY session's artifact
        dir are blocked — including the current session's, by design.
      - everything else: normal handler invocation.

    Known limitation (2C-B1.4): this is "path-based tool-layer
    isolation". bash is NOT guarded here because its ``command``
    argument is arbitrary shell; bash isolation is deferred to 2C-B2.
    Until 2C-B2 lands, a session could in principle ``cat`` another
    session's artifact via bash if it knew the FS path — the
    ``artifact://`` URI scheme ensures the model never learns that path
    from us, but a determined model could guess ``.harness/artifacts/``.
    The conclusion is therefore "path-based-tool-layer isolation",
    NOT "filesystem isolation".
    """
    # read_file artifact:// URI routing.
    if block_name == "read_file":
        _read_path = effective_args.get("path", "")
        if isinstance(_read_path, str) and _read_path.startswith("artifact://"):
            try:
                _data = call_store.read_by_uri(_read_path)
                return _data.decode("utf-8", errors="replace")
            except Exception as e:
                return f"Error: Cannot read artifact: {e}"

    # Stage 2C-B3B: grep_search / glob_search must NOT receive an
    # ``artifact://`` URI as their ``path`` argument. Artifact URIs are
    # a read_file-only channel; grep/glob cannot search inside an
    # artifact. Reject explicitly and tell the model to use read_file
    # with the URI instead. Without this, the URI would be treated as a
    # relative filesystem path, silently fail to resolve, and produce a
    # confusing "path not found" error that hides the real problem.
    if block_name in _PATH_BASED_TOOLS and block_name != "read_file":
        _path_arg = effective_args.get("path", "")
        if isinstance(_path_arg, str) and _path_arg.startswith("artifact://"):
            return ("Error: artifact:// URIs are not supported as a "
                    "search path for this tool. Use read_file with the "
                    "artifact:// URI to view the artifact content, or "
                    "search a workspace directory instead.")

    # Path-based guard: deny any path that resolves into artifact_root.
    if block_name in _PATH_BASED_TOOLS:
        _path_arg = effective_args.get("path", "")
        if _path_resolves_into_artifact_root(_path_arg):
            return ("Error: Access denied. Artifact access requires "
                    "artifact:// URI (for read_file) or must target a "
                    "path outside the artifact directory. Direct "
                    "filesystem paths into the artifact directory are "
                    "blocked.")

    # Normal execution path for all tools and all non-guarded paths.
    return _invoke_handler(
        block_name, effective_args, active_handlers, tool_profile,
        call_registry=call_registry,
    )


class SecureSandboxError(RuntimeError):
    """Raised when ``secure_multi_session`` mode rejects the sandbox or
    artifact store configuration at agent_loop startup.

    Stage 2C-B2B-3/4: this is a fail-fast startup error, not a runtime
    denial. The model never sees a bash tool it can't safely use —
    either the configuration passes and bash is exposed, or the
    configuration fails and agent_loop raises before the first model
    request.

    Path-leak contract (B2B-4): ``str(exception)`` returns a
    PUBLIC message that does NOT contain physical artifact paths or
    mount source paths — this is what may appear in model context,
    user-facing logs, or remote traces. The detailed reasons (which DO
    contain paths for operator diagnosis) are kept in
    ``diagnostic_reasons`` and should only be written to local debug
    logs or surfaced to the operator console, never to the model.
    """

    def __init__(
        self,
        public_message: str,
        *,
        diagnostic_reasons: tuple[str, ...] = (),
    ) -> None:
        super().__init__(public_message)
        self.public_message = public_message
        self.diagnostic_reasons = diagnostic_reasons

    def __str__(self) -> str:
        # str() returns ONLY the public message — no physical paths.
        # Operators who need the details access .diagnostic_reasons
        # directly and are responsible for not leaking them to the
        # model.
        return self.public_message


def _validate_secure_sandbox(
    *,
    sandbox,
    artifact_root: Path,
    active_tool_names: list[str],
) -> None:
    """Stage 2C-B2B-3/4: fail-fast startup validation for secure mode.

    Called by agent_loop when ``RUN_MODE == "secure_multi_session"`` and
    the active tool set contains ``bash``. Performs two checks:

    1. Capability check: the sandbox must *support* filesystem
       isolation (``capabilities.supports_filesystem_isolation``).
       NoOpSandbox fails here.
    2. Runtime assessment: ``sandbox.assess_isolation()`` evaluates the
       *actual* mount plan against the artifact root. A DockerSandbox
       that mounts a host directory containing the artifact root fails
       here, even though its capability flag is True.

    Raises ``SecureSandboxError`` on failure. The public message never
    contains physical paths; the detailed reasons (with paths) are in
    ``exception.diagnostic_reasons`` for operator diagnosis only.

    Note: this check only covers the DEFAULT artifact root. If the
    caller injected a custom ``artifact_store`` via agent_loop's
    keyword arg, agent_loop passes that store's ``root_dir`` instead.
    """
    if "bash" not in active_tool_names:
        # bash is not in the active profile — nothing to validate.
        # Other tools (read_file etc.) have their own path-based
        # guard and do not depend on sandbox filesystem isolation.
        return

    caps = getattr(sandbox, "capabilities", None)
    if caps is None or not caps.supports_filesystem_isolation:
        backend_name = type(sandbox).__name__
        raise SecureSandboxError(
            "secure_multi_session mode requires a sandbox with "
            "supports_filesystem_isolation=True, but the active "
            f"backend ({backend_name}) does not support it. Use "
            "DockerSandbox or switch to trusted_local mode.",
            diagnostic_reasons=(
                f"backend={backend_name}, supports_filesystem_isolation="
                f"{getattr(caps, 'supports_filesystem_isolation', None)}",
            ),
        )

    assessment = sandbox.assess_isolation(
        workdir=str(WORKDIR),
        private_paths=(str(artifact_root),),
    )
    if not assessment.filesystem_isolated:
        # Public message: no paths. Diagnostic reasons: full paths for
        # the operator. The model never sees diagnostic_reasons.
        raise SecureSandboxError(
            "secure_multi_session mode rejected: the current sandbox "
            "configuration does NOT isolate the artifact root from "
            "bash. Move the artifact root outside all mounted host "
            "directories, or use a sandbox with a stricter mount plan.",
            diagnostic_reasons=assessment.reasons,
        )


def agent_loop(messages: list, event_callback=None, tool_profile: str | None = None,
               session_id: str | None = None, *, artifact_store=None,
               tool_contributors=None):
    """Run the agent loop.

    Parameters
    ----------
    messages : list
        Conversation history.
    event_callback : callable, optional
        Callback for streaming events.
    tool_profile : str | None, optional (stage 2B)
        If None (default), all registered tools are exposed and executable —
        identical to pre-2B behavior. If a known profile name ("coding",
        "planning", "readonly", "team"), only the profile's whitelisted tools
        are sent to the model AND only those handlers are executable. An
        unknown profile name raises UnknownToolProfileError before the first
        model request. Profile is per-call: concurrent agent_loop() runs with
        different profiles do not interfere.

        Snapshot semantics: active_tools and active_handlers are resolved
        ONCE at agent_loop() startup. Tools registered or unregistered DURING
        this call do not appear until the NEXT agent_loop() call. This makes
        the active set stable for the entire run and avoids mid-loop tool
        set churn.

    session_id : str | None, optional (stage 2C-B1.3)
        Session ID for artifact isolation. If None, an id of the form
        ``run-<uuid16>`` is generated for THIS call so that concurrent
        agent_loop() invocations never share an artifact directory even
        when both omit session_id. If provided, the caller owns the
        stable identity (e.g. a session manager) and is responsible for
        not reusing it across unrelated runs.

        Per-call isolation contract (2C-B1.4):
        - Stable across the whole agent_loop() call: YES (one store, one
          session_id, one policy).
        - Shared across different agent_loop() calls: NO. Each call gets
          its own store/policy instance. There is no module-level
          default session for agent_loop to fall back to.
        - Cross-session artifact access via path-based tools
          (read_file/write_file/edit_file/grep_search/glob_search):
          rejected at the tool layer (see
          ``_execute_tool_or_artifact_read``). Paths are unconditionally
          normalized before containment is checked, so backslash, ``..``,
          and symlink variants do not bypass the guard.
        - bash is NOT yet guarded (its ``command`` arg is arbitrary
          shell). bash artifact isolation is the responsibility of 2C-B2.
          The current conclusion is therefore "path-based-tool-layer
          isolation", NOT "filesystem isolation".

    artifact_store : ArtifactStore, optional (keyword-only)
        If provided, use this pre-built store instead of constructing a
        fresh one. The store's session_id becomes the effective session_id
        for this call (overriding ``session_id``). Useful for tests that
        need to wrap the store (e.g. tracking/broken store) and for a
        future external session manager that pools stores. The store
        must already be bound to ``_ARTIFACT_ROOT``.

    tool_contributors : sequence | None, optional (keyword-only, stage 2D-A/2D-B)
        None (default) → use ``DEFAULT_TOOL_CONTRIBUTORS`` (TodoExtension +
        TaskExtension + SubagentExtension), so the default behavior still
        exposes TodoWrite, the four task tools, and the subagent ``task``
        tool, and the model sees the same 25 tools in the same order as
        pre-2D-B.
        () (empty) → explicitly disable ALL optional extensions; TodoWrite,
        the task tools, and ``task`` are then unknown to the model (calling
        them returns "Unknown tool").
        (ext1, ext2, ...) → enable ONLY the listed contributors; there is
        NO implicit merge with defaults — callers wanting "defaults +
        custom" must compose explicitly (e.g. ``(*DEFAULT_TOOL_CONTRIBUTORS,
        custom)``).
        Each contributor implements the ``ToolContributor`` protocol
        (duck-typed: any object with a ``contribute_tools(registry)``
        method). Contributor tools go into a per-call ``ToolRegistryOverlay``
        over ``BASE_TOOL_REGISTRY`` — the Base registry is NEVER mutated.
        Concurrent agent_loop() calls with different contributors do not
        interfere. Contributor tools are visible to the model only when the
        active profile's whitelist includes them (or ``tool_profile=None``).
        Conflict with a Base tool name raises ValueError at startup
        (fail-fast, before any model request). A contributor that raises
        mid-registration also fails before the model request; the per-call
        overlay is discarded, so Base and later agent_loop() calls are
        unaffected.

    event_callback : callable, optional
        Called with ``{"type": "tool_call"|"tool_result"|"text"|"status", ...}``
        dicts for real-time streaming.  When *None* (default) the function
        behaves exactly as before (prints to stdout only).

    Extension hooks (stage 1)
    -------------------------
    8 hook points are wired via ``EXTENSIONS.emit``. When no extensions are
    registered (default), each emit() returns an empty DispatchOutcome and the
    loop behaves identically to pre-stage-1. Kernel safety (dangerous-command
    blocking) stays in base_tools.run_bash and is NOT moved into extensions.

    AGENT_END contract
    ------------------
    AGENT_END fires EXACTLY ONCE per agent_loop() invocation, in a finally
    block. It fires regardless of whether the loop completed normally, was
    blocked by an extension, raised an exception, or was cancelled. The
    context includes ``status`` and ``error``.

    Block level distinction:
    - Agent-level block (BEFORE_AGENT_START / BEFORE_MODEL_REQUEST returns
      block=True): agent_status = "blocked", loop exits immediately.
    - Tool-level block (BEFORE_TOOL_CALL returns block=True): the single
      tool call is denied, a denied result is recorded, but agent_status
      is NOT changed. The model can continue with other tools or produce
      a final answer. The agent may still end with status="completed".

    Cancellation handling:
    - asyncio.CancelledError: status="cancelled" (caught before Exception)
    - KeyboardInterrupt: status="cancelled"
    Both are re-raised so the caller sees them.

    Error masking:
    - AGENT_END emit() is wrapped in try/except. If a cleanup handler fails,
      the failure is logged but does NOT mask the original exception from
      the try block. AGENT_END is effectively fail-open.
    """
    # Stage 2B: resolve active tools/handlers for THIS call's profile.
    # profile=None means all tools (pre-2B behavior). Unknown profile raises
    # before any model request — fail loud, never silently degrade to "all".
    #
    # Stage 2D-A: build a per-call overlay that combines Base tools with
    # Extension-contributed tools. The Base registry is NEVER mutated.
    # Stage 2D-B/2D-C/2D-D1: tool_contributors=None (default) →
    # DEFAULT_TOOL_CONTRIBUTORS (TodoExtension + TaskExtension +
    # SubagentExtension), so the default behavior still exposes TodoWrite +
    # the four task tools + the subagent ``task`` tool.
    # tool_contributors=() explicitly disables ALL optional extensions —
    # TodoWrite, task tools, and ``task`` are then unknown to the model. A
    # non-empty sequence enables ONLY the listed contributors (no implicit
    # merge with defaults); callers wanting "defaults + custom" must compose
    # explicitly.
    if tool_contributors is None:
        tool_contributors = DEFAULT_TOOL_CONTRIBUTORS
    _call_registry = ToolRegistryOverlay(BASE_TOOL_REGISTRY)
    for _contributor in tool_contributors:
        _contribute = getattr(_contributor, "contribute_tools", None)
        if _contribute is not None:
            _contribute(_call_registry)
    active_tools = _call_registry.resolve(profile=tool_profile)
    active_handlers = _call_registry.resolve_handlers(profile=tool_profile)

    # Stage 2C-B1.4: per-call session isolation for artifacts. Every
    # agent_loop() call gets a unique session_id unless the caller provides
    # a stable one. This ensures concurrent runs NEVER share artifact
    # directories, even when both omit session_id.
    if artifact_store is not None:
        _call_store = artifact_store
        _effective_session_id = getattr(_call_store, "_session_id", session_id)
    else:
        _effective_session_id = session_id or f"run-{_uuid.uuid4().hex[:16]}"
        _call_store = ArtifactStore(
            root_dir=_ARTIFACT_ROOT,
            session_id=_effective_session_id,
        )
    _call_policy = ToolOutputPolicy(
        store=_call_store,
        config=OutputPolicyConfig(),
        workdir=WORKDIR,
        artifact_root=_ARTIFACT_ROOT,
        session_id=_effective_session_id,
    )
    _call_session_id = _effective_session_id

    # Stage 2C-B2B-3/4.1: fail-fast startup validation for secure mode.
    # If the active profile exposes bash and RUN_MODE is secure_multi_session,
    # verify the sandbox actually isolates the artifact root from bash before
    # the first model request. This catches misconfigured Docker mounts AND
    # injected artifact_stores whose root lies inside a mounted directory.
    # trusted_local mode skips this check entirely.
    #
    # B2B-4.1/B2B-4.2: on success, set a per-Task secure-bash context
    # (ContextVar) bound to the current sandbox instance. run_bash()
    # checks this context in secure mode; without it (direct call from
    # outside agent_loop, or sandbox swapped after validation), bash
    # fails closed. ContextVar is per-asyncio-Task, not per-thread, so
    # concurrent Tasks on the same event-loop thread do NOT share the
    # context. reset(token) in finally restores the outer context
    # (supports nested agent_loop calls).
    #
    # B2B-4.2: the context also carries a nonce registered in a
    # process-wide live set. reset() discards the nonce, so any child
    # Task that inherited a COPY of the context (via asyncio.create_task)
    # cannot use it after this agent_loop ends — closing the child-Task
    # inheritance bypass.
    _secure_context_cv_token = None
    if RUN_MODE == "secure_multi_session":
        _active_tool_names = [t.get("name", "") for t in active_tools
                              if isinstance(t, dict)]
        _validate_secure_sandbox(
            sandbox=SANDBOX,
            artifact_root=_call_store.root_dir,
            active_tool_names=_active_tool_names,
        )
        # Validation passed — set the per-Task context so run_bash
        # accepts calls from this agent_loop() invocation. The token
        # binds to id(SANDBOX) so swapping the global SANDBOX after
        # validation invalidates the context. The nonce ensures child
        # Tasks cannot use a copied context after this agent_loop resets.
        _secure_context_cv_token = set_secure_bash_context(
            run_id=_effective_session_id,
            sandbox=SANDBOX,
        )

    # HOOK 1/8: BEFORE_AGENT_START
    EXTENSIONS.emit(Event.BEFORE_AGENT_START, {
        "event": Event.BEFORE_AGENT_START,
        "messages": messages,
    })

    agent_status = "completed"
    agent_error = None
    try:
        rounds_without_todo = 0
        while True:
            # s06: compression pipeline
            microcompact(messages)
            if estimate_tokens(messages) > TOKEN_THRESHOLD:
                print("[auto-compact triggered]")
                if event_callback:
                    event_callback({"type": "status", "message": "auto-compact triggered"})
                # HOOK 8/8: BEFORE_COMPACTION (auto)
                EXTENSIONS.emit(Event.BEFORE_COMPACTION, {
                    "event": Event.BEFORE_COMPACTION,
                    "pre_compact_messages": messages,
                })
                messages[:] = auto_compact(messages)
            # s08: drain background notifications
            notifs = BG.drain()
            if notifs:
                txt = "\n".join(
                    f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"<background-results>\n{txt}\n</background-results>",
                    }
                )
            # s10: check lead inbox
            inbox = BUS.read_inbox("lead")
            if inbox:
                messages.append(
                    {"role": "user", "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>"}
                )

            # HOOK 2/8: BEFORE_MODEL_REQUEST
            model_request_kwargs = {
                "model": MODEL,
                "system": SYSTEM,
                "messages": messages,
                "tools": active_tools,
                "max_tokens": 8000,
            }
            pre_model_outcome = EXTENSIONS.emit(Event.BEFORE_MODEL_REQUEST, {
                "event": Event.BEFORE_MODEL_REQUEST,
                "model": MODEL,
                "system_prompt": SYSTEM,
                "tools": active_tools,
                "messages": messages,
                "request_kwargs": model_request_kwargs,
            })
            # Apply patches (no-op when registry empty)
            if pre_model_outcome.model_request_patch:
                model_request_kwargs.update(pre_model_outcome.model_request_patch)
            if pre_model_outcome.blocked:
                # An extension blocked the model call. End the loop.
                agent_status = "blocked"
                agent_error = pre_model_outcome.block_reason
                return

            # LLM call (Phase 3A-1: via AnthropicAdapter; errors pass
            # through unwrapped, D-3).
            response = ANTHROPIC_ADAPTER.complete(ModelRequest(
                model=model_request_kwargs.get("model", MODEL),
                messages=model_request_kwargs.get("messages", messages),
                tools=model_request_kwargs.get("tools"),
                system=model_request_kwargs.get("system"),
                max_tokens=model_request_kwargs.get("max_tokens"),
                temperature=model_request_kwargs.get("temperature"),
                # 3A bridge: extension-patched request kwargs without a
                # unified field reach the client verbatim.
                metadata={
                    k: v for k, v in model_request_kwargs.items()
                    if k not in (
                        "model", "messages", "tools", "system",
                        "max_tokens", "temperature",
                    )
                },
            ))
            messages.append({"role": "assistant", "content": response.raw_response.content})

            # HOOK 3/8: AFTER_MODEL_RESPONSE
            post_model_outcome = EXTENSIONS.emit(Event.AFTER_MODEL_RESPONSE, {
                "event": Event.AFTER_MODEL_RESPONSE,
                "response": response,
                "messages": messages,
            })
            # Allow extension to substitute response (rare); default no-op.
            if post_model_outcome.model_response_patch is not None:
                response = post_model_outcome.model_response_patch

            # Emit events for streaming frontend
            if event_callback:
                # Token usage event (from ModelResponse.usage, if present)
                usage = response.usage
                if usage is not None:
                    in_tok = usage.input_tokens
                    out_tok = usage.output_tokens
                    event_callback({
                        "type": "tokens",
                        "input": in_tok,
                        "output": out_tok,
                        "tokens": in_tok + out_tok,
                    })
                if response.text:
                    event_callback({"type": "text", "text": response.text})
                for tc in response.tool_calls:
                    event_callback({
                        "type": "tool_call",
                        "name": tc.name,
                        "id": tc.id,
                        "input": tc.arguments,
                    })

            if response.stop_reason != StopReason.TOOL_CALL:
                # Loop ending normally. AGENT_END fires in finally.
                return
            # Tool execution
            results = []
            used_todo = False
            manual_compress = False
            for tc in response.tool_calls:
                if tc.name == "compress":
                    manual_compress = True

                # HOOK 4/8: BEFORE_TOOL_CALL
                tool_ctx = {
                    "event": Event.BEFORE_TOOL_CALL,
                    "tool_name": tc.name,
                    "tool_use_id": tc.id,
                    "tool_args": dict(tc.arguments),
                    "actor": "lead",
                }
                pre_tool_outcome = EXTENSIONS.emit(Event.BEFORE_TOOL_CALL, tool_ctx)
                # Apply tool_args_patch (no-op when empty)
                effective_args = dict(tc.arguments)
                if pre_tool_outcome.tool_args_patch:
                    effective_args.update(pre_tool_outcome.tool_args_patch)

                if pre_tool_outcome.blocked:
                    # TOOL-LEVEL BLOCK: this single tool call is denied.
                    # The agent is NOT marked "blocked" — the model can
                    # still continue reasoning with other tools or
                    # produce a final text answer. Only BEFORE_AGENT_START
                    # and BEFORE_MODEL_REQUEST blocks set agent_status to
                    # "blocked" (Agent-level block).
                    output = f"Blocked by extension: {pre_tool_outcome.block_reason or 'no reason'}"
                    print(f"> {tc.name}: BLOCKED")
                else:
                    # Stage 2C-B1.4: Artifact-root guard for path-based
                    # tools. read_file additionally supports artifact://
                    # URI routing. See ``_execute_tool_or_artifact_read``
                    # for the full contract and known limitation.
                    output = _execute_tool_or_artifact_read(
                        tc.name,
                        effective_args,
                        active_handlers,
                        tool_profile,
                        call_store=_call_store,
                        call_session_id=_call_session_id,
                        call_registry=_call_registry,
                    )
                    print(f"> {tc.name}:")
                    print(str(output)[:200])
                    if event_callback:
                        event_callback({
                            "type": "tool_result",
                            "name": tc.name,
                            "id": tc.id,
                            "output": str(output)[:2000],
                        })

                    # Stage 2C-B1: apply ToolOutputPolicy for large outputs.
                    # Runs BEFORE AFTER_TOOL_RESULT so extensions see the
                    # already-truncated, artifact-offloaded content, never
                    # the raw huge output. Only applied to tools in
                    # _OUTPUT_POLICY_TOOLS and only on successful execution
                    # (not for blocked/unknown/error/inactive results).
                    # Stage 2C-B2A: BashExecutionResult with non-zero exit
                    # code is NOT a transport error — it still goes through
                    # policy so exit_code is preserved in metadata.
                    if (tc.name in _OUTPUT_POLICY_TOOLS
                            and not pre_tool_outcome.blocked
                            and not _is_tool_error(output)):
                        _processed = _call_policy.process(
                            tool_name=tc.name,
                            raw_result=output,
                            context={
                                "tool_use_id": tc.id,
                                "tool_args": tc.arguments,
                            },
                        )
                        output = _processed.content

                    # HOOK 5/8: AFTER_TOOL_RESULT
                    post_tool_outcome = EXTENSIONS.emit(Event.AFTER_TOOL_RESULT, {
                        "event": Event.AFTER_TOOL_RESULT,
                        "tool_name": tc.name,
                        "tool_use_id": tc.id,
                        "tool_result": output,
                        "tool_error": output if _is_tool_error(output) else None,
                    })
                    # Allow extension to rewrite the tool result content
                    if post_tool_outcome.tool_result_patch is not None:
                        output = post_tool_outcome.tool_result_patch.get("content", output)

                    # Stage 2C-B1.2: Final output guard — Kernel-level hard
                    # limit for ALL tools, not just _OUTPUT_POLICY_TOOLS.
                    # This is a separate concern from the per-tool OutputPolicy:
                    #   - OutputPolicy: tool-specific Artifact + semantic preview
                    #   - FinalOutputGuard: global context-size hard limit
                    # Even tools not yet wired to OutputPolicy (bash, grep)
                    # get emergency hard-truncation here. Extensions cannot
                    # bypass this.
                    output = _call_policy.enforce_final(
                        tool_name=tc.name,
                        content=output,
                        context={"tool_use_id": tc.id},
                    )

                results.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": str(output)}
                )
                if tc.name == "TodoWrite":
                    used_todo = True
            # s03: nag reminder (only when todo workflow is active)
            rounds_without_todo = 0 if used_todo else rounds_without_todo + 1
            if TODO.has_open_items() and rounds_without_todo >= 3:
                results.append(
                    {"type": "text", "text": "<reminder>Update your todos.</reminder>"}
                )
            messages.append({"role": "user", "content": results})

            # HOOK 6/8: TURN_END
            EXTENSIONS.emit(Event.TURN_END, {
                "event": Event.TURN_END,
                "messages": messages,
            })

            # s06: manual compress
            if manual_compress:
                print("[manual compact]")
                if event_callback:
                    event_callback({"type": "status", "message": "manual compact"})
                # HOOK 8/8: BEFORE_COMPACTION (manual)
                EXTENSIONS.emit(Event.BEFORE_COMPACTION, {
                    "event": Event.BEFORE_COMPACTION,
                    "pre_compact_messages": messages,
                })
                messages[:] = auto_compact(messages)
                # Manual compress returns from loop. AGENT_END fires in finally.
                return
    except asyncio.CancelledError:
        # Async cancellation. Must be caught BEFORE Exception (it inherits
        # from BaseException in Python 3.8+). Re-raise so the caller's
        # event loop sees the cancellation.
        agent_status = "cancelled"
        agent_error = "asyncio.CancelledError"
        raise
    except KeyboardInterrupt:
        # Sync cancellation (Ctrl-C in REPL). Re-raise so the caller sees it.
        agent_status = "cancelled"
        agent_error = "KeyboardInterrupt"
        raise
    except Exception as exc:
        agent_status = "failed"
        agent_error = str(exc)
        raise
    finally:
        # HOOK 7/8: AGENT_END — fires EXACTLY ONCE per agent_loop() call,
        # regardless of normal completion, extension block, exception, or
        # cancellation. Extensions use this for trace finalization, resource
        # cleanup, and session state persistence.
        #
        # CRITICAL: AGENT_END handler failures must NOT mask the original
        # exception from the try block. We wrap emit() in try/except and log
        # any failure. This means AGENT_END is effectively fail-open: if a
        # cleanup handler raises, we log it but still re-raise the original
        # exception (if any).
        try:
            EXTENSIONS.emit(Event.AGENT_END, {
                "event": Event.AGENT_END,
                "messages": messages,
                "status": agent_status,
                "error": agent_error,
            })
        except Exception:
            logging.getLogger("agents.harness_core").exception(
                "AGENT_END hook failed; original agent_status=%s", agent_status
            )
        # B2B-4.1: reset the secure-bash context so it cannot leak past
        # this agent_loop() call. Uses ContextVar.reset(token) (NOT
        # set(None)) so that a nested agent_loop that set its own context
        # restores the outer context on exit — clearing only its own
        # frame, not clobbering a concurrently-running outer agent_loop.
        # A subsequent direct run_bash() in this Task will be rejected
        # until another validated agent_loop() sets a fresh context.
        #
        # B2B-4.2: reset() ALSO discards the nonce from the live set,
        # so any child Task that inherited a copy of the context can
        # no longer use it — closing the child-Task inheritance bypass.
        if _secure_context_cv_token is not None:
            reset_secure_bash_context(_secure_context_cv_token)


# === SECTION: repl ============================================================
if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36mharness >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/compact":
            if history:
                print("[manual compact via /compact]")
                history[:] = auto_compact(history)
            continue
        if query.strip() == "/tasks":
            print(TASK_MGR.list_all())
            continue
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
