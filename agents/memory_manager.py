"""memory_manager.py — Persistent file-based memory for the agent harness.

Memories are grouped by day: one ``.memory/YYYY-MM-DD.md`` file per day,
each containing one or more memory sections.  A ``MEMORY.md`` index tracks
every memory name → file#section for quick listing.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any

import yaml


_VALID_MEMORY_TYPES: set[str] = {"preference", "user_preference", "long_term", "protected"}


class MemoryManager:
    """CRUD manager for persistent agent memories stored in daily .md files.

    File format (per day)::

        ---
        date: 2026-06-09
        ---

        ## memory-name

        ```yaml
        memory_type: long_term
        description: one-line summary
        ```

        <body content>

        ---

        ## another-memory
        ...
    """

    SECTION_DIVIDER = "\n\n---\n\n"

    def __init__(self, memory_dir: Path) -> None:
        self._dir = memory_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "MEMORY.md"
        if not self._index_path.exists():
            self._index_path.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        content: str,
        *,
        name: str = "",
        description: str = "",
        memory_type: str = "long_term",
    ) -> str:
        """Create a new memory.  Auto-slugs *name* from *content* if empty."""
        name = self._resolve_name(name, content)
        if not name:
            return "Error: memory name is invalid. Use only letters, digits, dots, underscores, and dashes (max 64 chars)."
        if not self._name_is_safe(name):
            return f"Error: memory name {name!r} is invalid. Use only letters, digits, dots, underscores, and dashes (max 64 chars)."

        memory_type = self._normalize_memory_type(memory_type)
        if memory_type not in _VALID_MEMORY_TYPES:
            return f"Error: invalid memory_type {memory_type!r}. Must be one of: {', '.join(sorted(_VALID_MEMORY_TYPES))}."

        # If name already exists anywhere, delegate to update
        existing_file = self._file_for_name(name)
        if existing_file is not None:
            return self.update(name, content=content, description=description, memory_type=memory_type)

        # Append to today's daily file
        day_file = self._today_file()
        sections = self._parse_daily_file(day_file)
        sections[name] = {
            "content": content.strip(),
            "description": description or self._summarize(content),
            "memory_type": memory_type,
        }
        self._write_daily_file(day_file, sections)
        self._index_upsert(name, sections[name]["description"], self._today_date())
        return f"Created memory {name!r} (type: {memory_type})."

    def update(
        self,
        name: str,
        *,
        content: str | None = None,
        description: str | None = None,
        memory_type: str | None = None,
    ) -> str:
        """Update an existing memory, merging the given fields."""
        day_file = self._file_for_name(name)
        if day_file is None:
            return f"Error: memory {name!r} not found. Use memory_recall to list existing memories."

        sections = self._parse_daily_file(day_file)
        if name not in sections:
            return f"Error: memory {name!r} not found. Use memory_recall to list existing memories."

        entry = sections[name]
        if content is not None:
            entry["content"] = content.strip()
        if description is not None:
            entry["description"] = description
        elif content is not None and not entry.get("description"):
            entry["description"] = self._summarize(content)
        if memory_type is not None:
            mt = self._normalize_memory_type(memory_type)
            if mt not in _VALID_MEMORY_TYPES:
                return f"Error: invalid memory_type {memory_type!r}. Must be one of: {', '.join(sorted(_VALID_MEMORY_TYPES))}."
            entry["memory_type"] = mt

        self._write_daily_file(day_file, sections)
        self._index_upsert(name, entry["description"], self._date_from_filename(day_file))
        return f"Updated memory {name!r}."

    def delete(self, name: str) -> str:
        """Delete a memory by name."""
        day_file = self._file_for_name(name)
        if day_file is None:
            return f"Error: memory {name!r} not found."

        sections = self._parse_daily_file(day_file)
        if name not in sections:
            return f"Error: memory {name!r} not found."

        del sections[name]

        if sections:
            self._write_daily_file(day_file, sections)
        else:
            # No more sections — remove the daily file
            day_file.unlink(missing_ok=True)

        self._index_remove(name)
        return f"Deleted memory {name!r}."

    def recall(self, *, query: str = "", name: str = "") -> str:
        """Search or list memories.

        - ``name`` given: return the full content of that memory.
        - ``query`` given: keyword search across names, descriptions, and bodies.
        - neither: list all memories (name + description).
        """
        if name:
            day_file = self._file_for_name(name)
            if day_file is None:
                return f"Error: memory {name!r} not found."
            sections = self._parse_daily_file(day_file)
            if name not in sections:
                return f"Error: memory {name!r} not found."
            return sections[name]["content"]

        all_memories = self._list_all()
        if not all_memories:
            return "(no memories)"

        if not query:
            lines = [f"- {e['name']}: {e['description']}" for e in all_memories]
            return "\n".join(lines) if lines else "(no memories)"

        # Keyword search
        q = query.lower()
        matches = []
        for entry in all_memories:
            score = 0
            if q in entry["name"].lower():
                score += 10
            if q in entry["description"].lower():
                score += 5
            if q in entry.get("body", "").lower():
                score += 2
            if score > 0:
                matches.append((score, entry))

        if not matches:
            return f"No memories match {query!r}."

        matches.sort(key=lambda item: (-item[0], item[1]["name"]))
        lines = []
        for score, entry in matches:
            body_preview = self._truncate(entry.get("body", ""), 200)
            lines.append(
                f"## {entry['name']}  (score={score})\n"
                f"  {entry['description']}\n"
                f"  {body_preview}"
            )
        return "\n".join(lines)

    def load_all_as_messages(self) -> list[dict[str, Any]]:
        """Read every memory and return them as protected message dicts.

        Protection is purely via ``metadata.memory_type`` field.
        No XML tag wrapping — ``token_budget.is_protected_memory_message``
        checks the metadata path.
        """
        messages: list[dict[str, Any]] = []
        for entry in self._list_all():
            mem_type = entry.get("memory_type", "long_term")
            content = f"## {entry['name']}\n\n{entry['body']}"
            messages.append({
                "role": "user",
                "content": content,
                "metadata": {
                    "memory_type": mem_type,
                    "memory_name": entry["name"],
                },
            })

        _order = {"protected": 0, "long_term": 1, "user_preference": 2, "preference": 2}
        messages.sort(key=lambda m: _order.get(m["metadata"]["memory_type"], 3))
        return messages

    def inject_into_messages(self, messages: list[dict[str, Any]]) -> None:
        """Prepend current memory set to messages.

        Previous injected-memory messages are **kept** so the model can see
        the evolution of memories over time.
        """
        fresh = self.load_all_as_messages()
        for mem_msg in reversed(fresh):
            messages.insert(0, mem_msg)

    # ------------------------------------------------------------------
    # Daily file read / write
    # ------------------------------------------------------------------

    def _today_file(self) -> Path:
        return self._dir / f"{self._today_date()}.md"

    @staticmethod
    def _today_date() -> str:
        return datetime.date.today().isoformat()

    @staticmethod
    def _date_from_filename(filepath: Path) -> str:
        return filepath.stem  # "2026-06-09"

    def _file_for_name(self, name: str) -> Path | None:
        """Find which daily file (if any) contains *name*."""
        for filepath in sorted(self._dir.glob("*.md"), reverse=True):
            if filepath.name == "MEMORY.md":
                continue
            sections = self._parse_daily_file(filepath)
            if name in sections:
                return filepath
        return None

    def _parse_daily_file(self, filepath: Path) -> dict[str, dict[str, str]]:
        """Parse a daily file into {name: {content, description, memory_type}}.

        Returns an empty dict if the file doesn't exist.
        """
        if not filepath.exists():
            return {}

        raw = filepath.read_text(encoding="utf-8")
        # Strip file-level frontmatter
        body = self._strip_frontmatter_body(raw)

        sections: dict[str, dict[str, str]] = {}
        # Split on the section divider
        chunks = body.split(self.SECTION_DIVIDER)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            name, entry = self._parse_section(chunk)
            if name and entry:
                sections[name] = entry
        return sections

    def _write_daily_file(self, filepath: Path, sections: dict[str, dict[str, str]]) -> None:
        """Write a daily file from sections dict."""
        date_str = self._date_from_filename(filepath)
        lines = [
            "---",
            f"date: {date_str}",
            "---",
            "",
        ]

        # Sort sections for stable output (preserve insertion order roughly by name)
        sorted_names = sorted(sections.keys())
        for i, name in enumerate(sorted_names):
            entry = sections[name]
            if i > 0:
                lines.append("")
                lines.append("---")
                lines.append("")

            lines.append(f"## {name}")
            lines.append("")

            # Inline YAML metadata block
            meta = {
                "memory_type": entry.get("memory_type", "long_term"),
                "description": entry.get("description", ""),
            }
            lines.append("```yaml")
            lines.append(yaml.dump(meta, default_flow_style=False, allow_unicode=True).strip())
            lines.append("```")
            lines.append("")
            lines.append(entry["content"].strip())

        lines.append("")  # trailing newline
        filepath.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # Section parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_section(chunk: str) -> tuple[str | None, dict[str, str] | None]:
        """Parse one memory section into (name, {content, description, memory_type})."""
        # Expect:  ## name\n\n```yaml\n...\n```\n\n<body>
        match = re.match(r"^##\s+(.+?)\s*\n+(.*)", chunk, re.DOTALL)
        if not match:
            return None, None

        name = match.group(1).strip()
        remainder = match.group(2).strip()

        # Try to pull out the YAML metadata block
        yaml_match = re.match(r"```yaml\s*\n(.*?)\n```\s*\n(.*)", remainder, re.DOTALL)
        if yaml_match:
            try:
                meta = yaml.safe_load(yaml_match.group(1)) or {}
            except yaml.YAMLError:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            body = yaml_match.group(2).strip()
        else:
            meta = {}
            body = remainder

        entry = {
            "content": body,
            "description": str(meta.get("description", "")),
            "memory_type": str(meta.get("memory_type", "long_term")),
        }
        return name, entry

    # ------------------------------------------------------------------
    # Index (MEMORY.md)
    # ------------------------------------------------------------------

    def _index_upsert(self, name: str, description: str, date_str: str) -> None:
        description = self._truncate(description, 120)
        line = f"- [{name}]({date_str}.md) — {description}"
        lines = self._read_index_lines()
        prefix = f"- [{name}]"
        replaced = False
        new_lines = []
        for existing in lines:
            if existing.strip().startswith(prefix):
                new_lines.append(line)
                replaced = True
            else:
                new_lines.append(existing)
        if not replaced:
            new_lines.append(line)
        self._write_index_lines(new_lines)

    def _index_remove(self, name: str) -> None:
        prefix = f"- [{name}]"
        lines = [ln for ln in self._read_index_lines() if not ln.strip().startswith(prefix)]
        self._write_index_lines(lines)

    def _read_index_lines(self) -> list[str]:
        if self._index_path.exists():
            text = self._index_path.read_text(encoding="utf-8")
            return [ln for ln in text.splitlines() if ln.strip()]
        return []

    def _write_index_lines(self, lines: list[str]) -> None:
        self._index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # List all
    # ------------------------------------------------------------------

    def _list_all(self) -> list[dict[str, Any]]:
        """Return {name, description, memory_type, body} for every memory."""
        entries: list[dict[str, Any]] = []
        for filepath in sorted(self._dir.glob("*.md")):
            if filepath.name == "MEMORY.md":
                continue
            sections = self._parse_daily_file(filepath)
            for name, entry in sections.items():
                entries.append({
                    "name": name,
                    "description": entry.get("description", ""),
                    "memory_type": entry.get("memory_type", "long_term"),
                    "body": entry["content"],
                })
        return entries

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_frontmatter_body(text: str) -> str:
        match = re.match(r"^---\s*\n.*?\n---\s*\n(.*)", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    @staticmethod
    def _summarize(text: str) -> str:
        cleaned = re.sub(r"\*\*|\*|__|_|`|#", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        sentence = re.split(r"[.。!?！？\n]", cleaned)[0].strip()
        return sentence[:120] if sentence else cleaned[:120]

    @staticmethod
    def _slug(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_]+", "-", text)
        text = re.sub(r"-{2,}", "-", text)
        text = text.strip("-")
        return text[:64] or "memory"

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        return text if len(text) <= max_chars else text[:max_chars - 3] + "..."

    @staticmethod
    def _name_is_safe(name: str) -> bool:
        return bool(re.fullmatch(r"[a-zA-Z0-9._-]{1,64}", name))

    # ------------------------------------------------------------------
    # Name resolution + type aliasing
    # ------------------------------------------------------------------

    def _resolve_name(self, name: str, content: str) -> str:
        if name.strip():
            if not self._name_is_safe(name.strip()):
                return ""
            return self._slug(name.strip())
        return self._slug(self._summarize(content))

    @staticmethod
    def _normalize_memory_type(raw: str) -> str:
        raw_lower = raw.strip().lower()
        aliases: dict[str, str] = {
            "preference": "preference",
            "user_preference": "user_preference",
            "user_preferences": "user_preference",
            "long_term": "long_term",
            "long_term_memory": "long_term",
            "protected": "protected",
        }
        return aliases.get(raw_lower, raw_lower)
