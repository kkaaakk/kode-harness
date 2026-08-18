"""
tool_registry.py - Tool registry with registered/active separation.

Stage 2B scope:
- ToolRegistry holds all REGISTERED tools (the universe).
- PROFILE DEFINITIONS are explicit whitelists of tool names, kept separate
  from tool registration. A tool does NOT declare which profiles it belongs
  to; instead, each profile declares which tool names it allows.
- resolve(profile) returns the ACTIVE set: registered tools whose names
  appear in the profile whitelist, in REGISTRATION order (not profile order).
- profile=None means "no profile filtering" = all visible tools. This is the
  default and preserves pre-2B behavior exactly.
- Unknown profiles raise UnknownToolProfileError (never silently degrade to
  "all tools" — that would be a permission risk).

Registry holds "what tools exist". Profile + resolve() decides "what's
active this turn". The agent_loop caller passes tool_profile per-call, so
concurrent agents with different profiles don't interfere.

Stage 2D-A scope:
- ToolEntry now carries ``owner`` and ``source`` attribution so conflict
  errors can name both sides (e.g. "extension 'todo' conflicts with
  extension 'custom-todo'").
- ``ToolRegistryOverlay`` provides a per-call read-only-on-parent view:
  local additions/overrides do NOT mutate the parent Base registry, so
  concurrent agent_loop() calls with different Extension tool sets do
  not pollute each other.
- ``ToolContributor`` Protocol allows Extensions to contribute tools to
  a per-call overlay without modifying the global registry.
- Profile resolution is tolerant of whitelisted-but-uninstalled optional
  tools: a profile that lists ``TodoWrite`` still resolves successfully
  when the Todo Extension is disabled — the missing tool is simply
  absent from the active set. Only an unknown PROFILE name raises.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class UnknownToolProfileError(ValueError):
    """Raised when resolve() is called with a profile name that is not
    defined in STANDARD_PROFILES.

    This fails loud and early (before the model request) rather than
    silently degrading to "all tools", which would be a permission risk
    (e.g. a typo "readnoly" must NOT expose write tools).
    """


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@dataclass
class ToolEntry:
    """A registered tool."""
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Any]
    # Whether the tool is exposed to the model by default.
    # Hidden tools (e.g. idle, claim_task) are registered but not in TOOLS.
    visible: bool = True
    # Permission hint for kernel safety / tracing. Not enforced here.
    permission: str = "default"  # "shell" | "file_write" | "default"
    # Stage 2D-A: attribution for conflict reporting and tracing.
    # owner: who owns this tool — "kernel" for builtins, or an
    #        extension_id (e.g. "todo-extension") for contributed tools.
    # source: where this tool came from — "builtin" | "extension" | "mcp".
    owner: str = "kernel"
    source: str = "builtin"
    # Stage 2D-B: stable sort key. resolve() emits tools ordered by
    # (order ASC, registration_order ASC) so an Extension-contributed tool
    # can occupy a specific position in the active set (e.g. TodoWrite keeps
    # its legacy slot) instead of being appended at the end. Default 0 keeps
    # pre-2D-B behavior for tools that don't care about position.
    order: int = 0


# ---------------------------------------------------------------------------
# Standard profiles — explicit whitelists of tool NAMES.
# Order here does NOT matter; resolve() emits in (order ASC, registration
# order ASC) — see ToolEntry.order (stage 2D-B).
# Tool names MUST match the actual registered names (e.g. "read_file" not
# "read", "bash" not "run_bash").
# ---------------------------------------------------------------------------

STANDARD_PROFILES: dict[str, tuple[str, ...]] = {
    "coding": (
        "read_file",
        "write_file",
        "edit_file",
        "bash",
        "grep_search",
        "glob_search",
    ),
    "planning": (
        "read_file",
        "grep_search",
        "glob_search",
        "TodoWrite",
        "task_create",
        "task_get",
        "task_update",
        "task_list",
    ),
    "readonly": (
        "read_file",
        "grep_search",
        "glob_search",
    ),
    "team": (
        "read_file",
        "write_file",
        "edit_file",
        "bash",
        "grep_search",
        "glob_search",
        "task",
        "spawn_teammate",
        "list_teammates",
        "send_message",
        "read_inbox",
        "broadcast",
        "shutdown_request",
        "plan_approval",
    ),
}

# Human-readable descriptions (kept for backwards compat with stage 1 tests
# that import PROFILES).
PROFILES = {
    "coding": "Full coding workflow: read/write/edit/bash/grep/glob",
    "planning": "Read-only analysis + task management",
    "readonly": "Read-only: read/grep/glob only",
    "team": "Multi-agent: coding + subagent + team_manager",
}


class ToolRegistry:
    """Registry of tools with profile-based active-set resolution.

    Thread-safe. The registry stores the UNIVERSE of registered tools.
    Profile-based filtering happens in resolve() at call time, so different
    callers can use different profiles concurrently without mutating shared
    state.
    """

    def __init__(self, profiles: dict[str, tuple[str, ...]] | None = None):
        self._tools: dict[str, ToolEntry] = {}
        self._lock = threading.RLock()
        # Profile definitions are immutable references; resolve() reads them
        # at call time. Defaults to STANDARD_PROFILES.
        self._profiles: dict[str, tuple[str, ...]] = (
            dict(profiles) if profiles is not None else dict(STANDARD_PROFILES)
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: Callable[..., Any],
        *,
        visible: bool = True,
        permission: str = "default",
        overwrite: bool = False,
        owner: str = "kernel",
        source: str = "builtin",
        order: int = 0,
    ) -> None:
        """Register a tool.

        Args:
            name: Tool name (must be unique unless overwrite=True).
            description: Human-readable description shown to the model.
            input_schema: JSON schema dict for tool input.
            handler: Callable invoked with tool kwargs.
            visible: If False, registered but not exposed to the model.
            permission: Hint for kernel safety ("shell" | "file_write" | ...).
            overwrite: If False (default), raise ValueError on duplicate name.
                       Production wiring should keep overwrite=False to catch
                       silent overwrites (e.g. MCP tool shadowing a builtin).
                       Tests or hot-reload may set overwrite=True.
            owner: Stage 2D-A attribution — who owns this tool
                   ("kernel" for builtins, extension_id for extensions).
            source: Stage 2D-A attribution — "builtin" | "extension" | "mcp".
            order: Stage 2D-B stable sort key. resolve() emits active tools
                   ordered by (order ASC, registration_order ASC) so a
                   contributed tool can keep a specific slot instead of being
                   appended at the end. Default 0.

        Raises:
            ValueError: if name already registered and overwrite=False.
                The error message names BOTH owners so conflict is clear.
        """
        with self._lock:
            existing = self._tools.get(name)
            if existing is not None and not overwrite:
                raise ValueError(
                    f"Tool '{name}' from owner '{owner}' (source='{source}') "
                    f"conflicts with existing tool owned by "
                    f"'{existing.owner}' (source='{existing.source}'). "
                    f"Use overwrite=True to replace, or unregister first."
                )
            entry = ToolEntry(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=handler,
                visible=visible,
                permission=permission,
                owner=owner,
                source=source,
                order=order,
            )
            self._tools[name] = entry

    def unregister(self, name: str) -> bool:
        with self._lock:
            return self._tools.pop(name, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._tools.clear()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolEntry | None:
        with self._lock:
            return self._tools.get(name)

    def get_handler(self, name: str) -> Callable[..., Any] | None:
        """Return the handler for a tool, or None if not registered.

        This is the Registry equivalent of ``TOOL_HANDLERS.get(name)``.
        Returns None (not raising) so callers can produce the same
        "Unknown tool: xxx" message as before.
        """
        with self._lock:
            entry = self._tools.get(name)
            return entry.handler if entry else None

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    def all_names(self) -> list[str]:
        with self._lock:
            return sorted(self._tools.keys())

    def all_entries(self) -> list[ToolEntry]:
        with self._lock:
            return list(self._tools.values())

    def known_profile(self, profile: str) -> bool:
        """Return True if profile is a defined profile name."""
        return profile in self._profiles

    def is_active(self, name: str, profile: str | None) -> bool:
        """Return True if `name` is registered AND active under `profile`.

        - profile=None: active iff registered (and visible).
        - profile=<name>: active iff registered, visible, AND in the
          profile's whitelist.
        - Unknown profile: returns False (callers should check
          known_profile() first to raise a clear error).
        """
        with self._lock:
            entry = self._tools.get(name)
            if entry is None or not entry.visible:
                return False
            if profile is None:
                return True
            allowed = self._profiles.get(profile)
            if allowed is None:
                return False
            return name in allowed

    # ------------------------------------------------------------------
    # Active-set resolution
    # ------------------------------------------------------------------

    def resolve(self, profile: str | None = None) -> list[dict]:
        """Return the list of tool schemas to expose to the model.

        Args:
            profile: If None, return all visible tools (no filtering).
                     If a known profile name, return visible tools whose
                     names are in the profile's whitelist.
                     If an UNKNOWN profile name, raise UnknownToolProfileError.

        Returns:
            List of {"name", "description", "input_schema"} dicts, in
            REGISTRATION order (not profile whitelist order), ready to pass
            to provider.messages.create(tools=...).
        """
        if profile is not None and profile not in self._profiles:
            raise UnknownToolProfileError(
                f"Unknown tool profile: {profile!r}. "
                f"Known profiles: {sorted(self._profiles.keys())}"
            )
        with self._lock:
            entries = list(self._tools.values())
            allowed = set(self._profiles[profile]) if profile is not None else None

        # Stage 2D-B: stable order — (order ASC, registration_order ASC).
        # registration_order is the original insertion index in self._tools
        # (dict preserves insertion order in Python 3.7+); it acts as the
        # tiebreaker when two tools share the same ``order``.
        entries = [e for _, e in sorted(
            enumerate(entries), key=lambda i_e: (i_e[1].order, i_e[0])
        )]
        result = []
        for entry in entries:
            if not entry.visible:
                continue
            if allowed is not None and entry.name not in allowed:
                continue
            result.append({
                "name": entry.name,
                "description": entry.description,
                "input_schema": entry.input_schema,
            })
        return result

    def resolve_handlers(self, profile: str | None = None) -> dict[str, Callable[..., Any]]:
        """Return {name: handler} for the active set.

        Same filtering as resolve(). Keys are in registration order.
        """
        if profile is not None and profile not in self._profiles:
            raise UnknownToolProfileError(
                f"Unknown tool profile: {profile!r}. "
                f"Known profiles: {sorted(self._profiles.keys())}"
            )
        with self._lock:
            entries = list(self._tools.values())
            allowed = set(self._profiles[profile]) if profile is not None else None

        # Stage 2D-B: stable order — (order ASC, registration_order ASC).
        entries = [e for _, e in sorted(
            enumerate(entries), key=lambda i_e: (i_e[1].order, i_e[0])
        )]
        result = {}
        for entry in entries:
            if not entry.visible:
                continue
            if allowed is not None and entry.name not in allowed:
                continue
            result[entry.name] = entry.handler
        return result


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

default_registry = ToolRegistry()


# ---------------------------------------------------------------------------
# Stage 2D-A: ToolContributor protocol + ToolRegistryOverlay
# ---------------------------------------------------------------------------

class ToolContributor(Protocol):
    """Protocol for Extensions that contribute tools to a per-call overlay.

    An Extension does NOT have to implement this protocol. The runtime
    uses ``getattr(extension, "contribute_tools", None)`` to detect
    contributors; non-contributors are silently skipped.

    The protocol intentionally takes a ``ToolRegistry`` (or
    ``ToolRegistryOverlay``) argument rather than returning a list, so
    the contributor can call ``registry.register(...)`` with proper
    ``owner``/``source`` attribution and let the registry enforce
    conflict detection (fail-fast on duplicate names).
    """

    extension_id: str

    def contribute_tools(self, registry: "ToolRegistry") -> None:
        ...


class ToolRegistryOverlay(ToolRegistry):
    """Per-call overlay over a Base ToolRegistry.

    Stage 2D-A: provides a view that combines Base tools with local
    additions/overrides, WITHOUT mutating the parent. This allows
    concurrent agent_loop() calls to enable different Extension tool
    sets without polluting each other.

    Semantics:
        - Reads (get/has/resolve/resolve_handlers) see Base tools +
          local overrides. Local entries take precedence on name
          collision (i.e. an Extension can override a Base tool's
          handler/schema if explicitly allowed — but 2D-A does NOT
          allow this; ``overwrite=False`` is enforced at register time
          for Base names).
        - Writes (register/unregister/clear) only affect the local
          overlay; the parent is NEVER modified.
        - Registration order: Base tools first (in parent order), then
          local tools (in overlay insertion order). This matches the
          pre-2D-A behavior where builtins come first.

    Conflict handling:
        - Local-vs-Base name collision: raise ValueError at register
          time (fail-fast). The error names BOTH owners.
        - Local-vs-Local name collision: same fail-fast behavior.

    Thread safety:
        - The overlay has its own lock for local state. Parent reads
          use the parent's lock. The two locks are never held
          simultaneously in a way that could deadlock (resolve reads
          parent first, releases, then reads local).
    """

    def __init__(
        self,
        parent: ToolRegistry,
        profiles: dict[str, tuple[str, ...]] | None = None,
    ):
        super().__init__(profiles=profiles)
        # ``_tools`` inherited from parent holds LOCAL entries only.
        # We keep a reference to the parent for reads.
        self._parent = parent

    # ------------------------------------------------------------------
    # Reads — merge parent + local
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolEntry | None:
        local = super().get(name)
        if local is not None:
            return local
        return self._parent.get(name)

    def get_handler(self, name: str) -> Callable[..., Any] | None:
        entry = self.get(name)
        return entry.handler if entry else None

    def has(self, name: str) -> bool:
        return super().has(name) or self._parent.has(name)

    def all_names(self) -> list[str]:
        # Local names take precedence on collision; parent names fill in.
        seen = set(super().all_names())
        merged = list(super().all_names())
        for n in self._parent.all_names():
            if n not in seen:
                merged.append(n)
                seen.add(n)
        return sorted(merged)

    def all_entries(self) -> list[ToolEntry]:
        # Parent entries first (in parent order), then local entries
        # that don't shadow a parent name. Local entries that DO shadow
        # a parent name replace the parent entry IN PLACE to preserve
        # registration order. (2D-A does not allow shadowing Base names
        # by default, but the overlay supports it for future use.)
        parent_entries = self._parent.all_entries()
        local_entries = super().all_entries()
        local_names = {e.name for e in local_entries}
        result: list[ToolEntry] = []
        for entry in parent_entries:
            if entry.name in local_names:
                # Find the local override (fail-fast at register time
                # ensures at most one local entry per name).
                local_match = next(
                    e for e in local_entries if e.name == entry.name
                )
                result.append(local_match)
            else:
                result.append(entry)
        # Append local entries that don't shadow any parent name.
        for entry in local_entries:
            if entry.name not in {e.name for e in parent_entries}:
                result.append(entry)
        return result

    def known_profile(self, profile: str) -> bool:
        return super().known_profile(profile) or self._parent.known_profile(profile)

    def is_active(self, name: str, profile: str | None) -> bool:
        entry = self.get(name)
        if entry is None or not entry.visible:
            return False
        if profile is None:
            return True
        # Check local profiles first, then parent profiles.
        if profile in self._profiles:
            return name in self._profiles[profile]
        if self._parent.known_profile(profile):
            return name in self._parent._profiles[profile]
        return False

    # ------------------------------------------------------------------
    # Active-set resolution — merge parent + local, filter by profile
    # ------------------------------------------------------------------

    def resolve(self, profile: str | None = None) -> list[dict]:
        """Return tool schemas for the active set.

        Stage 2D-A: merges parent + local entries (local overrides on
        name collision), then filters by profile. Profile resolution is
        TOLERANT of whitelisted-but-uninstalled optional tools: a
        profile that lists ``TodoWrite`` still resolves successfully
        when the Todo Extension is disabled — the missing tool is
        simply absent from the active set. Only an unknown PROFILE name
        raises UnknownToolProfileError.
        """
        # Validate profile name against either local or parent profiles.
        if profile is not None:
            if (profile not in self._profiles
                    and not self._parent.known_profile(profile)):
                raise UnknownToolProfileError(
                    f"Unknown tool profile: {profile!r}. "
                    f"Known profiles: {sorted(self._profiles.keys())}"
                )
        entries = self.all_entries()
        if profile is None:
            allowed = None
        elif profile in self._profiles:
            allowed = set(self._profiles[profile])
        else:
            allowed = set(self._parent._profiles[profile])

        # Stage 2D-B: stable order — (order ASC, registration_order ASC).
        # all_entries() returns parent-first + local-appended (with local
        # shadowing parent in place); that ordering is the
        # registration_order tiebreaker here, so a contributed tool with a
        # specific ``order`` lands in its intended slot instead of at the end.
        entries = [e for _, e in sorted(
            enumerate(entries), key=lambda i_e: (i_e[1].order, i_e[0])
        )]
        result = []
        for entry in entries:
            if not entry.visible:
                continue
            # Tolerant filtering: if the tool is in the whitelist, include
            # it; if not, skip. Missing optional tools are simply absent
            # from ``entries`` so they never reach here — no error.
            if allowed is not None and entry.name not in allowed:
                continue
            result.append({
                "name": entry.name,
                "description": entry.description,
                "input_schema": entry.input_schema,
            })
        return result

    def resolve_handlers(self, profile: str | None = None) -> dict[str, Callable[..., Any]]:
        """Return {name: handler} for the active set (parent + local)."""
        if profile is not None:
            if (profile not in self._profiles
                    and not self._parent.known_profile(profile)):
                raise UnknownToolProfileError(
                    f"Unknown tool profile: {profile!r}. "
                    f"Known profiles: {sorted(self._profiles.keys())}"
                )
        entries = self.all_entries()
        if profile is None:
            allowed = None
        elif profile in self._profiles:
            allowed = set(self._profiles[profile])
        else:
            allowed = set(self._parent._profiles[profile])

        # Stage 2D-B: stable order — (order ASC, registration_order ASC).
        entries = [e for _, e in sorted(
            enumerate(entries), key=lambda i_e: (i_e[1].order, i_e[0])
        )]
        result = {}
        for entry in entries:
            if not entry.visible:
                continue
            if allowed is not None and entry.name not in allowed:
                continue
            result[entry.name] = entry.handler
        return result

    # ------------------------------------------------------------------
    # Registration — fail-fast on collision with parent
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: Callable[..., Any],
        *,
        visible: bool = True,
        permission: str = "default",
        overwrite: bool = False,
        owner: str = "kernel",
        source: str = "builtin",
        order: int = 0,
    ) -> None:
        """Register a tool in the OVERLAY only (parent is never modified).

        Stage 2D-A: fail-fast if the name already exists in the parent
        Base registry OR in the local overlay, unless ``overwrite=True``.
        The error message names BOTH owners so the conflict source is
        clear. 2D-A does NOT allow Extensions to override Base tools —
        callers should keep ``overwrite=False`` for production wiring.
        """
        # Check parent first (without holding the parent lock for long).
        parent_entry = self._parent.get(name)
        if parent_entry is not None and not overwrite:
            raise ValueError(
                f"Tool '{name}' from owner '{owner}' (source='{source}') "
                f"conflicts with existing BASE tool owned by "
                f"'{parent_entry.owner}' (source='{parent_entry.source}'). "
                f"Extensions may not override Base tools in 2D-A. "
                f"Use overwrite=True only for tested replacement scenarios."
            )
        # Then check local + insert via parent implementation.
        super().register(
            name, description, input_schema, handler,
            visible=visible, permission=permission,
            overwrite=overwrite, owner=owner, source=source, order=order,
        )


# ---------------------------------------------------------------------------
# Stage 2D-B.1: LegacyToolRegistryView — read-only public compatibility view.
# ---------------------------------------------------------------------------


class LegacyToolRegistryView:
    """Read-only public view over the default composed registry.

    Stage 2D-B.1: ``TOOL_REGISTRY`` is this view, NOT the mutable
    ``ToolRegistryOverlay``. External code can iterate tool names, index
    entries by name, resolve profiles, and query handlers — but CANNOT
    ``register`` / ``unregister`` / ``clear`` (those raise ``TypeError`` to
    direct callers to ``BASE_TOOL_REGISTRY`` for mutation or
    ``build_default_tool_registry()`` for a fresh composed overlay).

    This fixes the stage 2A compatibility debt where ``set(TOOL_REGISTRY)``
    and ``TOOL_REGISTRY[name]`` raised ``TypeError`` because neither
    ``ToolRegistry`` nor ``ToolRegistryOverlay`` implemented ``__iter__`` /
    ``__getitem__``. The view provides a stable, intention-revealing public
    API instead of overloading the overlay with ambiguous iteration semantics
    just to satisfy one legacy test.

    Reads delegate to the source (the default composed overlay), so the view
    reflects the default tool set (Base + DEFAULT_TOOL_CONTRIBUTORS). It does
    NOT hold its own snapshot — the source is treated as immutable after
    import (``agent_loop`` builds fresh per-call overlays and never mutates
    the module-level composed registry).

    Public API contract (fixed by 2D-B.1):
      * ``set(view)`` / ``iter(view)`` → registered tool NAMES (str)
      * ``view["bash"]`` → ``ToolEntry`` (KeyError if absent)
      * ``len(view)`` → number of registered tools
      * ``"bash" in view`` → bool
      * ``view.all_names()`` / ``view.get()`` / ``view.get_handler()`` /
        ``view.resolve()`` / ``view.resolve_handlers()`` → delegated
      * ``view.register()`` / ``view.unregister()`` / ``view.clear()``
        → ``TypeError`` (read-only)
    """

    __slots__ = ("_source",)

    def __init__(self, source: ToolRegistry) -> None:
        self._source = source

    # --- iteration / mapping protocol (read-only) ---
    def __iter__(self):
        # Iterate over ALL registered tool names (visible or not), matching
        # the pre-2D-A meaning of "the registry's tools".
        return iter(self._source.all_names())

    def __len__(self) -> int:
        return len(self._source.all_names())

    def __contains__(self, name: str) -> bool:
        return self._source.has(name)

    def __getitem__(self, name: str) -> ToolEntry:
        entry = self._source.get(name)
        if entry is None:
            raise KeyError(name)
        return entry

    # --- delegated read access ---
    def all_names(self) -> list[str]:
        return self._source.all_names()

    def all_entries(self) -> list[ToolEntry]:
        return self._source.all_entries()

    def get(self, name: str):
        return self._source.get(name)

    def get_handler(self, name: str):
        return self._source.get_handler(name)

    def has(self, name: str) -> bool:
        return self._source.has(name)

    def resolve(self, profile=None):
        return self._source.resolve(profile=profile)

    def resolve_handlers(self, profile=None):
        return self._source.resolve_handlers(profile=profile)

    # --- mutation explicitly forbidden ---
    def _read_only(self, op: str):
        raise TypeError(
            f"TOOL_REGISTRY is a read-only legacy view (stage 2D-B.1): "
            f"{op}() is not allowed. Use BASE_TOOL_REGISTRY for mutation, "
            f"or build_default_tool_registry() for a fresh composed overlay."
        )

    def register(self, *args, **kwargs):
        self._read_only("register")

    def unregister(self, *args, **kwargs):
        self._read_only("unregister")

    def clear(self, *args, **kwargs):
        self._read_only("clear")


__all__ = [
    "ToolEntry",
    "ToolRegistry",
    "ToolRegistryOverlay",
    "LegacyToolRegistryView",
    "ToolContributor",
    "PROFILES",
    "STANDARD_PROFILES",
    "UnknownToolProfileError",
    "default_registry",
]
