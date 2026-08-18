"""test_todo_manager.py — TodoManager CRUD, validation, render tests.

Pure module: no external deps, no mocking needed.
"""

from __future__ import annotations

import pytest

from agents.todo_manager import TodoManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(content: str = "task", status: str = "pending", active_form: str = "doing") -> dict:
    return {"content": content, "status": status, "activeForm": active_form}


# ---------------------------------------------------------------------------
# CRUD basics
# ---------------------------------------------------------------------------


class TestTodoManagerCRUD:

    def test_initial_state_is_empty(self):
        mgr = TodoManager()
        assert mgr.items == []
        assert mgr.render() == "No todos."
        assert mgr.has_open_items() is False

    def test_update_adds_items(self):
        mgr = TodoManager()
        result = mgr.update([_item("Write tests"), _item("Fix bug")])
        assert len(mgr.items) == 2
        assert "Write tests" in result
        assert "Fix bug" in result

    def test_update_replaces_all_items(self):
        mgr = TodoManager()
        mgr.update([_item("Old task")])
        mgr.update([_item("New task")])
        assert len(mgr.items) == 1
        assert mgr.items[0]["content"] == "New task"

    def test_status_persisted_correctly(self):
        mgr = TodoManager()
        mgr.update([
            _item("done", status="completed"),
            _item("wip", status="in_progress"),
            _item("todo", status="pending"),
        ])
        assert mgr.items[0]["status"] == "completed"
        assert mgr.items[1]["status"] == "in_progress"
        assert mgr.items[2]["status"] == "pending"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestTodoManagerValidation:

    def test_empty_content_rejected(self):
        mgr = TodoManager()
        with pytest.raises(ValueError, match="content required"):
            mgr.update([{"content": "", "status": "pending", "activeForm": "x"}])

    def test_missing_content_rejected(self):
        mgr = TodoManager()
        with pytest.raises(ValueError, match="content required"):
            mgr.update([{"status": "pending", "activeForm": "x"}])

    def test_invalid_status_rejected(self):
        mgr = TodoManager()
        with pytest.raises(ValueError, match="invalid status"):
            mgr.update([_item(status="blocked")])

    def test_missing_active_form_rejected(self):
        mgr = TodoManager()
        with pytest.raises(ValueError, match="activeForm required"):
            mgr.update([{"content": "task", "status": "pending", "activeForm": ""}])

    def test_max_20_todos_enforced(self):
        mgr = TodoManager()
        items = [_item(f"task-{i}") for i in range(21)]
        with pytest.raises(ValueError, match="Max 20"):
            mgr.update(items)

    def test_exactly_20_todos_allowed(self):
        mgr = TodoManager()
        items = [_item(f"task-{i}") for i in range(20)]
        mgr.update(items)
        assert len(mgr.items) == 20

    def test_only_one_in_progress_allowed(self):
        mgr = TodoManager()
        items = [
            _item("a", status="in_progress"),
            _item("b", status="in_progress"),
        ]
        with pytest.raises(ValueError, match="Only one in_progress"):
            mgr.update(items)

    def test_status_case_insensitive(self):
        mgr = TodoManager()
        mgr.update([_item(status="PENDING")])
        assert mgr.items[0]["status"] == "pending"

    def test_whitespace_content_stripped_and_rejected(self):
        mgr = TodoManager()
        with pytest.raises(ValueError, match="content required"):
            mgr.update([{"content": "   ", "status": "pending", "activeForm": "x"}])


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


class TestTodoManagerRender:

    def test_render_pending_marker(self):
        mgr = TodoManager()
        mgr.update([_item("task a")])
        assert "[ ] task a" in mgr.render()

    def test_render_in_progress_marker_and_active_form(self):
        mgr = TodoManager()
        mgr.update([_item("writing code", status="in_progress", active_form="Writing")])
        rendered = mgr.render()
        assert "[>] writing code" in rendered
        assert "<- Writing" in rendered

    def test_render_completed_marker(self):
        mgr = TodoManager()
        mgr.update([_item("done task", status="completed")])
        assert "[x] done task" in mgr.render()

    def test_render_progress_counter(self):
        mgr = TodoManager()
        mgr.update([
            _item("a", status="completed"),
            _item("b", status="completed"),
            _item("c"),
        ])
        assert "(2/3 completed)" in mgr.render()

    def test_render_empty(self):
        mgr = TodoManager()
        assert mgr.render() == "No todos."


# ---------------------------------------------------------------------------
# has_open_items
# ---------------------------------------------------------------------------


class TestTodoManagerHasOpenItems:

    def test_no_items_means_no_open(self):
        mgr = TodoManager()
        assert mgr.has_open_items() is False

    def test_all_completed_means_no_open(self):
        mgr = TodoManager()
        mgr.update([_item("a", status="completed"), _item("b", status="completed")])
        assert mgr.has_open_items() is False

    def test_one_pending_means_open(self):
        mgr = TodoManager()
        mgr.update([_item("a", status="completed"), _item("b")])
        assert mgr.has_open_items() is True

    def test_one_in_progress_means_open(self):
        mgr = TodoManager()
        mgr.update([_item("a", status="in_progress")])
        assert mgr.has_open_items() is True
