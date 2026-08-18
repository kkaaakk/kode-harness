"""test_task_manager.py — TaskManager CRUD, dependency graph, claim, list tests.

task_manager.py depends on agents.config.TASKS_DIR, which triggers heavy
deps (anthropic, dotenv). We mock config to isolate the module under test.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixture: load task_manager with a temp TASKS_DIR
# ---------------------------------------------------------------------------


@pytest.fixture()
def task_mgr(tmp_path: Path):
    """Yield a fresh TaskManager instance backed by tmp_path."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()

    # Build a fake config module so task_manager doesn't pull in anthropic/dotenv
    fake_config = types.ModuleType("agents.config")
    fake_config.TASKS_DIR = tasks_dir

    prev_config = sys.modules.get("agents.config")
    prev_task_mgr = sys.modules.get("agents.task_manager")
    sys.modules["agents.config"] = fake_config

    # Force-reload task_manager to pick up the fake TASKS_DIR
    if "agents.task_manager" in sys.modules:
        del sys.modules["agents.task_manager"]
    from agents.task_manager import TaskManager

    # Patch TASKS_DIR in the freshly loaded module (belt-and-suspenders)
    with patch("agents.task_manager.TASKS_DIR", tasks_dir):
        yield TaskManager()

    # Cleanup
    if prev_config is None:
        sys.modules.pop("agents.config", None)
    else:
        sys.modules["agents.config"] = prev_config
    if prev_task_mgr is not None:
        sys.modules["agents.task_manager"] = prev_task_mgr
    else:
        sys.modules.pop("agents.task_manager", None)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestTaskManagerCRUD:

    def test_create_returns_task_with_id(self, task_mgr):
        result = json.loads(task_mgr.create("Setup CI"))
        assert result["id"] == 1
        assert result["subject"] == "Setup CI"
        assert result["status"] == "pending"
        assert result["owner"] is None
        assert result["blockedBy"] == []

    def test_create_auto_increments_id(self, task_mgr):
        t1 = json.loads(task_mgr.create("Task A"))
        t2 = json.loads(task_mgr.create("Task B"))
        assert t1["id"] == 1
        assert t2["id"] == 2

    def test_create_with_description(self, task_mgr):
        result = json.loads(task_mgr.create("Deploy", description="Deploy to prod"))
        assert result["description"] == "Deploy to prod"

    def test_get_returns_task(self, task_mgr):
        created = json.loads(task_mgr.create("Read me"))
        fetched = json.loads(task_mgr.get(created["id"]))
        assert fetched["subject"] == "Read me"

    def test_get_nonexistent_raises(self, task_mgr):
        with pytest.raises(ValueError, match="not found"):
            task_mgr.get(999)

    def test_update_status(self, task_mgr):
        t = json.loads(task_mgr.create("Do it"))
        updated = json.loads(task_mgr.update(t["id"], status="in_progress"))
        assert updated["status"] == "in_progress"

    def test_delete_task(self, task_mgr):
        t = json.loads(task_mgr.create("Temp"))
        result = task_mgr.update(t["id"], status="deleted")
        assert "deleted" in result.lower()
        with pytest.raises(ValueError, match="not found"):
            task_mgr.get(t["id"])


# ---------------------------------------------------------------------------
# Dependency graph (blockedBy)
# ---------------------------------------------------------------------------


class TestTaskManagerDependencies:

    def test_add_blocked_by(self, task_mgr):
        t1 = json.loads(task_mgr.create("Prerequisite"))
        t2 = json.loads(task_mgr.create("Dependent"))
        updated = json.loads(task_mgr.update(t2["id"], add_blocked_by=[t1["id"]]))
        assert t1["id"] in updated["blockedBy"]

    def test_remove_blocked_by(self, task_mgr):
        t1 = json.loads(task_mgr.create("Blocker"))
        t2 = json.loads(task_mgr.create("Blocked"))
        task_mgr.update(t2["id"], add_blocked_by=[t1["id"]])
        updated = json.loads(task_mgr.update(t2["id"], remove_blocked_by=[t1["id"]]))
        assert t1["id"] not in updated["blockedBy"]

    def test_completing_task_unblocks_dependents(self, task_mgr):
        """Completing task A should remove A from all other tasks' blockedBy."""
        t1 = json.loads(task_mgr.create("Blocker"))
        t2 = json.loads(task_mgr.create("Dep 1"))
        t3 = json.loads(task_mgr.create("Dep 2"))

        task_mgr.update(t2["id"], add_blocked_by=[t1["id"]])
        task_mgr.update(t3["id"], add_blocked_by=[t1["id"]])

        # Complete the blocker
        task_mgr.update(t1["id"], status="completed")

        dep1 = json.loads(task_mgr.get(t2["id"]))
        dep2 = json.loads(task_mgr.get(t3["id"]))
        assert t1["id"] not in dep1["blockedBy"]
        assert t1["id"] not in dep2["blockedBy"]

    def test_blocked_by_deduplicates(self, task_mgr):
        t1 = json.loads(task_mgr.create("A"))
        t2 = json.loads(task_mgr.create("B"))
        task_mgr.update(t2["id"], add_blocked_by=[t1["id"]])
        task_mgr.update(t2["id"], add_blocked_by=[t1["id"]])
        result = json.loads(task_mgr.get(t2["id"]))
        assert result["blockedBy"].count(t1["id"]) == 1

    def test_multiple_blockers(self, task_mgr):
        t1 = json.loads(task_mgr.create("Blocker A"))
        t2 = json.loads(task_mgr.create("Blocker B"))
        t3 = json.loads(task_mgr.create("Dependent"))
        task_mgr.update(t3["id"], add_blocked_by=[t1["id"], t2["id"]])
        result = json.loads(task_mgr.get(t3["id"]))
        assert t1["id"] in result["blockedBy"]
        assert t2["id"] in result["blockedBy"]


# ---------------------------------------------------------------------------
# Claim (task assignment)
# ---------------------------------------------------------------------------


class TestTaskManagerClaim:

    def test_claim_sets_owner_and_status(self, task_mgr):
        t = json.loads(task_mgr.create("Claim me"))
        result = task_mgr.claim(t["id"], "agent-alpha")
        assert "agent-alpha" in result

        fetched = json.loads(task_mgr.get(t["id"]))
        assert fetched["owner"] == "agent-alpha"
        assert fetched["status"] == "in_progress"

    def test_claim_nonexistent_raises(self, task_mgr):
        with pytest.raises(ValueError, match="not found"):
            task_mgr.claim(999, "agent")

    def test_reclaim_overwrites_owner(self, task_mgr):
        t = json.loads(task_mgr.create("Shared"))
        task_mgr.claim(t["id"], "agent-a")
        task_mgr.claim(t["id"], "agent-b")
        fetched = json.loads(task_mgr.get(t["id"]))
        assert fetched["owner"] == "agent-b"


# ---------------------------------------------------------------------------
# list_all
# ---------------------------------------------------------------------------


class TestTaskManagerListAll:

    def test_list_empty(self, task_mgr):
        assert task_mgr.list_all() == "No tasks."

    def test_list_shows_all_tasks(self, task_mgr):
        task_mgr.create("Alpha")
        task_mgr.create("Beta")
        listing = task_mgr.list_all()
        assert "Alpha" in listing
        assert "Beta" in listing

    def test_list_shows_status_markers(self, task_mgr):
        t1 = json.loads(task_mgr.create("Pending"))
        t2 = json.loads(task_mgr.create("WIP"))
        task_mgr.update(t2["id"], status="in_progress")
        listing = task_mgr.list_all()
        assert "[ ]" in listing  # pending
        assert "[>]" in listing  # in_progress

    def test_list_shows_owner(self, task_mgr):
        t = json.loads(task_mgr.create("Owned"))
        task_mgr.claim(t["id"], "bob")
        listing = task_mgr.list_all()
        assert "@bob" in listing

    def test_list_shows_blocked_by(self, task_mgr):
        t1 = json.loads(task_mgr.create("Blocker"))
        t2 = json.loads(task_mgr.create("Blocked"))
        task_mgr.update(t2["id"], add_blocked_by=[t1["id"]])
        listing = task_mgr.list_all()
        assert "blocked by:" in listing

    def test_list_sorted_by_id(self, task_mgr):
        task_mgr.create("C")
        task_mgr.create("A")
        task_mgr.create("B")
        listing = task_mgr.list_all()
        lines = listing.strip().split("\n")
        # Should be sorted by file name (task_1, task_2, task_3)
        assert "#1" in lines[0]
        assert "#2" in lines[1]
        assert "#3" in lines[2]
