"""Unit tests for MemoryManager — CRUD, recall, injection message format.
Tests the daily-file storage model: all memories from one day go into
YYYY-MM-DD.md.
"""

from __future__ import annotations

import datetime
import tempfile
import unittest
from pathlib import Path

from agents.memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _daily_file(dir_path: Path, date_str: str = "") -> Path:
    date_str = date_str or datetime.date.today().isoformat()
    return dir_path / f"{date_str}.md"


def _memory_files(dir_path: Path) -> list[Path]:
    return sorted(
        f for f in dir_path.glob("*.md") if f.name != "MEMORY.md"
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

class MemoryManagerCRUDTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / ".memory"
        self.mgr = MemoryManager(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    # --- create ---

    def test_create_appends_to_todays_daily_file(self):
        result = self.mgr.create("User prefers dark mode.", name="user-prefers-dark-mode")
        self.assertIn("Created memory", result)

        daily = _daily_file(self.dir)
        self.assertTrue(daily.exists())
        raw = daily.read_text(encoding="utf-8")
        self.assertIn("## user-prefers-dark-mode", raw)
        self.assertIn("User prefers dark mode.", raw)

    def test_create_auto_slugs_name_from_content(self):
        result = self.mgr.create("The user likes Python type hints everywhere")
        self.assertIn("Created memory", result)
        self.assertNotIn("Error", result)

        daily = _daily_file(self.dir)
        raw = daily.read_text(encoding="utf-8")
        self.assertIn("python", raw.lower())

    def test_create_rejects_unsafe_name(self):
        result = self.mgr.create("content", name="not a safe name!")
        self.assertIn("Error", result)

    def test_create_rejects_invalid_memory_type(self):
        result = self.mgr.create("content", name="test", memory_type="bogus")
        self.assertIn("Error", result)

    def test_create_updates_existing_memory(self):
        self.mgr.create("Old content.", name="test-mem")
        result = self.mgr.create("New content.", name="test-mem")
        self.assertIn("Updated memory", result)

        daily = _daily_file(self.dir)
        raw = daily.read_text(encoding="utf-8")
        self.assertIn("New content.", raw)
        self.assertNotIn("Old content.", raw)

    def test_memory_only_creates_one_daily_file_for_same_day(self):
        self.mgr.create("Memory A", name="mem-a")
        self.mgr.create("Memory B", name="mem-b")
        self.assertEqual(1, len(_memory_files(self.dir)))

    def test_multiple_memories_in_same_daily_file(self):
        self.mgr.create("Content A.", name="mem-a", description="Desc A")
        self.mgr.create("Content B.", name="mem-b", description="Desc B")

        sections = self.mgr._parse_daily_file(_daily_file(self.dir))
        self.assertIn("mem-a", sections)
        self.assertIn("mem-b", sections)
        self.assertEqual("Content A.", sections["mem-a"]["content"])
        self.assertEqual("Content B.", sections["mem-b"]["content"])

    # --- update ---

    def test_update_merges_fields(self):
        self.mgr.create("Original body.", name="test-update", description="orig desc")
        result = self.mgr.update("test-update", content="New body.")
        self.assertIn("Updated memory", result)

        sections = self.mgr._parse_daily_file(_daily_file(self.dir))
        self.assertIn("New body.", sections["test-update"]["content"])
        self.assertEqual("orig desc", sections["test-update"]["description"])

    def test_update_not_found(self):
        result = self.mgr.update("nonexistent", content="x")
        self.assertIn("Error", result)

    # --- delete ---

    def test_delete_removes_section_from_daily_file(self):
        self.mgr.create("To be deleted.", name="delete-me")
        self.mgr.create("Keep me.", name="keep-me")
        self.assertTrue((_daily_file(self.dir)).exists())

        result = self.mgr.delete("delete-me")
        self.assertIn("Deleted memory", result)

        sections = self.mgr._parse_daily_file(_daily_file(self.dir))
        self.assertNotIn("delete-me", sections)
        self.assertIn("keep-me", sections)

    def test_delete_last_section_removes_daily_file(self):
        self.mgr.create("Only memory.", name="only")
        self.mgr.delete("only")
        self.assertFalse(_daily_file(self.dir).exists())

    def test_delete_not_found(self):
        result = self.mgr.delete("no-such-memory")
        self.assertIn("Error", result)

    # --- recall ---

    def test_recall_list_all(self):
        self.mgr.create("Fact A", name="fact-a", description="First fact")
        self.mgr.create("Fact B", name="fact-b", description="Second fact")

        result = self.mgr.recall()
        self.assertIn("fact-a", result)
        self.assertIn("fact-b", result)
        self.assertIn("First fact", result)
        self.assertIn("Second fact", result)

    def test_recall_by_name(self):
        self.mgr.create("Full content of the memory.", name="full-mem")
        result = self.mgr.recall(name="full-mem")
        self.assertIn("Full content of the memory.", result)
        self.assertNotIn("Error", result)

    def test_recall_by_name_not_found(self):
        result = self.mgr.recall(name="missing")
        self.assertIn("Error", result)

    def test_recall_by_query(self):
        self.mgr.create("alpha beta gamma", name="greek")
        self.mgr.create("delta epsilon", name="more-greek")

        result = self.mgr.recall(query="beta")
        self.assertIn("greek", result)
        self.assertNotIn("more-greek", result)

    def test_recall_empty(self):
        result = self.mgr.recall()
        self.assertIn("(no memories)", result)


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

class MemoryManagerInjectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / ".memory"
        self.mgr = MemoryManager(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_all_as_messages_sets_metadata(self):
        self.mgr.create("User prefers short answers.", name="short-answers",
                        memory_type="user_preference")
        self.mgr.create("Project uses Python 3.12+.", name="python-version",
                        memory_type="long_term")

        messages = self.mgr.load_all_as_messages()
        self.assertEqual(2, len(messages))

        for msg in messages:
            self.assertIsInstance(msg.get("metadata"), dict)
            self.assertIn(msg["metadata"]["memory_type"],
                          {"user_preference", "long_term"})
            self.assertTrue(
                isinstance(msg["metadata"].get("memory_name"), str)
                and len(msg["metadata"]["memory_name"]) > 0)

    def test_load_all_as_messages_no_xml_tags(self):
        """Protection is purely via metadata.memory_type — no XML tag wrapping."""
        self.mgr.create("Preference content.", name="pref-mem",
                        memory_type="user_preference")
        self.mgr.create("Long-term content.", name="lt-mem",
                        memory_type="long_term")

        messages = self.mgr.load_all_as_messages()
        for msg in messages:
            c = msg["content"]
            self.assertNotIn("<long_term_memory>", c)
            self.assertNotIn("<user_preferences>", c)
            self.assertNotIn("<user_preference>", c)
            self.assertNotIn("<identity>", c)
            self.assertIsInstance(msg.get("metadata"), dict)
            self.assertIn(msg["metadata"]["memory_type"],
                          {"user_preference", "long_term"})

    def test_inject_into_messages_prepends_and_replaces(self):
        self.mgr.create("Memory content.", name="mem-1")
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        self.mgr.inject_into_messages(messages)

        # First message should be the injected memory
        self.assertTrue(
            isinstance(messages[0].get("metadata", {}).get("memory_name"), str))
        self.assertIn("Memory content.", messages[0]["content"])

        # User message should still be there
        user_msgs = [m for m in messages if m.get("content") == "hello"]
        self.assertEqual(1, len(user_msgs))

    def test_inject_into_messages_preserves_old_versions(self):
        self.mgr.create("First version.", name="mem-1")
        messages = []
        self.mgr.inject_into_messages(messages)
        self.assertEqual(1, len(messages))

        # Update → inject again without removing old
        self.mgr.update("mem-1", content="Second version.")
        self.mgr.inject_into_messages(messages)

        mem_msgs = [m for m in messages
                    if isinstance(m.get("metadata"), dict)
                    and "memory_name" in m["metadata"]]
        self.assertEqual(2, len(mem_msgs))
        self.assertIn("Second version.", mem_msgs[0]["content"])
        self.assertIn("First version.", mem_msgs[1]["content"])

    def test_inject_leaves_non_memory_messages_untouched(self):
        self.mgr.create("A memory.", name="mem-a")
        messages = [
            {"role": "system", "content": "important system prompt"},
            {"role": "user", "content": "a regular question"},
        ]
        self.mgr.inject_into_messages(messages)

        sys_msgs = [m for m in messages if m.get("role") == "system"]
        self.assertEqual(1, len(sys_msgs))
        regular = [m for m in messages if m.get("content") == "a regular question"]
        self.assertEqual(1, len(regular))


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class MemoryManagerIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / ".memory"
        self.mgr = MemoryManager(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_memory_md_created_on_init(self):
        self.assertTrue((self.dir / "MEMORY.md").exists())

    def test_create_appends_to_index(self):
        self.mgr.create("Important fact.", name="important-fact",
                        description="Something important")
        index_text = (self.dir / "MEMORY.md").read_text(encoding="utf-8")
        self.assertIn("[important-fact]", index_text)
        self.assertIn("Something important", index_text)

    def test_delete_removes_from_index(self):
        self.mgr.create("Fact A.", name="fact-a")
        self.mgr.create("Fact B.", name="fact-b")
        self.mgr.delete("fact-a")

        index_text = (self.dir / "MEMORY.md").read_text(encoding="utf-8")
        self.assertNotIn("[fact-a]", index_text)
        self.assertIn("fact-b", index_text)


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

class MemoryManagerNormalizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / ".memory"
        self.mgr = MemoryManager(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_alias_user_preferences_to_user_preference(self):
        self.mgr.create("content", name="test", memory_type="user_preferences")
        sections = self.mgr._parse_daily_file(_daily_file(self.dir))
        self.assertEqual("user_preference", sections["test"]["memory_type"])

    def test_alias_long_term_memory_to_long_term(self):
        self.mgr.create("content", name="test2", memory_type="long_term_memory")
        sections = self.mgr._parse_daily_file(_daily_file(self.dir))
        self.assertEqual("long_term", sections["test2"]["memory_type"])


# ---------------------------------------------------------------------------
# Daily file format
# ---------------------------------------------------------------------------

class MemoryManagerDailyFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / ".memory"
        self.mgr = MemoryManager(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_daily_file_has_date_frontmatter(self):
        self.mgr.create("Content.", name="test")
        raw = _daily_file(self.dir).read_text(encoding="utf-8")
        self.assertIn("date:", raw)
        self.assertIn(datetime.date.today().isoformat(), raw)

    def test_section_has_yaml_metadata_block(self):
        self.mgr.create("Body text.", name="meta-test",
                        description="Test desc", memory_type="protected")
        raw = _daily_file(self.dir).read_text(encoding="utf-8")
        self.assertIn("```yaml", raw)
        self.assertIn("memory_type:", raw)
        self.assertIn("protected", raw)
        self.assertIn("description:", raw)
        self.assertIn("Test desc", raw)

    def test_parse_daily_file_roundtrip(self):
        self.mgr.create("Alpha.", name="a", description="First")
        self.mgr.create("Beta.", name="b", description="Second")

        sections = self.mgr._parse_daily_file(_daily_file(self.dir))
        self.assertEqual(2, len(sections))
        self.assertEqual("Alpha.", sections["a"]["content"])
        self.assertEqual("First", sections["a"]["description"])
        self.assertEqual("Beta.", sections["b"]["content"])

    def test_parse_daily_file_not_exists(self):
        result = self.mgr._parse_daily_file(self.dir / "2099-01-01.md")
        self.assertEqual({}, result)

    def test_cross_day_files_are_separate(self):
        # Manually write a memory in yesterday's file
        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        yesterday_file = self.dir / f"{yesterday}.md"
        self.mgr._write_daily_file(yesterday_file, {
            "old-mem": {
                "content": "Old content.",
                "description": "Old desc",
                "memory_type": "long_term",
            }
        })

        # Create a memory today
        self.mgr.create("Today content.", name="today-mem")

        # Both should exist as separate files
        self.assertTrue(yesterday_file.exists())
        self.assertTrue(_daily_file(self.dir).exists())

        # _list_all should return both
        all_entries = self.mgr._list_all()
        names = {e["name"] for e in all_entries}
        self.assertIn("old-mem", names)
        self.assertIn("today-mem", names)

        # _file_for_name should find each in its correct file
        self.assertEqual(yesterday_file, self.mgr._file_for_name("old-mem"))
        self.assertEqual(_daily_file(self.dir), self.mgr._file_for_name("today-mem"))


if __name__ == "__main__":
    unittest.main()
