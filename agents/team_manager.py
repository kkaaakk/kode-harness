"""
team_manager.py - Multi-agent team coordination.

Includes:
- ``MessageBus``: file-based async mailbox messaging
- ``TeammateManager``: persistent autonomous teammate lifecycle
- Shutdown / plan approval protocol helpers

Depends on: config, base_tools, task_manager
"""

import json
import threading
import time
import uuid
from pathlib import Path
from queue import Queue

from agents.config import (
    TEAM_DIR,
    INBOX_DIR,
    TASKS_DIR,
    POLL_INTERVAL,
    IDLE_TIMEOUT,
    WORKDIR,
    client,
    MODEL,
)
from agents.base_tools import run_bash, run_read, run_write, run_edit
from agents.task_manager import TaskManager


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------
class MessageBus:
    def __init__(self):
        INBOX_DIR.mkdir(parents=True, exist_ok=True)

    def send(
        self,
        sender: str,
        to: str,
        content: str,
        msg_type: str = "message",
        extra: dict = None,
    ) -> str:
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        with open(INBOX_DIR / f"{to}.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        path = INBOX_DIR / f"{name}.jsonl"
        if not path.exists():
            return []
        msgs = [json.loads(l) for l in path.read_text().strip().splitlines() if l]
        path.write_text("")
        return msgs

    def broadcast(self, sender: str, content: str, names: list) -> str:
        count = 0
        for n in names:
            if n != sender:
                self.send(sender, n, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


# ---------------------------------------------------------------------------
# Shutdown / plan tracking
# ---------------------------------------------------------------------------
shutdown_requests: dict = {}
plan_requests: dict = {}


def handle_shutdown_request(bus: "MessageBus", teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    bus.send(
        "lead", teammate, "Please shut down.", "shutdown_request", {"request_id": req_id}
    )
    return f"Shutdown request {req_id} sent to '{teammate}'"


def handle_plan_review(
    bus: "MessageBus", request_id: str, approve: bool, feedback: str = ""
) -> str:
    req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    req["status"] = "approved" if approve else "rejected"
    bus.send(
        "lead",
        req["from"],
        feedback,
        "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {req['status']} for '{req['from']}'"


# ---------------------------------------------------------------------------
# TeammateManager
# ---------------------------------------------------------------------------
class TeammateManager:
    def __init__(self, bus: MessageBus, task_mgr: TaskManager):
        TEAM_DIR.mkdir(parents=True, exist_ok=True)
        self.bus = bus
        self.task_mgr = task_mgr
        self.config_path = TEAM_DIR / "config.json"
        self.config = self._load()
        self.threads = {}

    def _load(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find(self, name: str) -> dict | None:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save()
        threading.Thread(
            target=self._loop, args=(name, role, prompt), daemon=True
        ).start()
        return f"Spawned '{name}' (role: {role})"

    def _set_status(self, name: str, status: str):
        member = self._find(name)
        if member:
            member["status"] = status
            self._save()

    def _loop(self, name: str, role: str, prompt: str):
        team_name = self.config["team_name"]
        sys_prompt = (
            f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
            f"Use idle when done with current work. You may auto-claim tasks."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = [
            {
                "name": "bash",
                "description": "Run command.",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
            {
                "name": "read_file",
                "description": "Read file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "edit_file",
                "description": "Edit file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            },
            {
                "name": "send_message",
                "description": "Send message.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["to", "content"],
                },
            },
            {
                "name": "idle",
                "description": "Signal no more work.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "claim_task",
                "description": "Claim task by ID.",
                "input_schema": {
                    "type": "object",
                    "properties": {"task_id": {"type": "integer"}},
                    "required": ["task_id"],
                },
            },
        ]
        while True:
            # -- WORK PHASE --
            for _ in range(50):
                inbox = self.bus.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
                    messages.append({"role": "user", "content": json.dumps(msg)})
                try:
                    response = client.messages.create(
                        model=MODEL,
                        system=sys_prompt,
                        messages=messages,
                        tools=tools,
                        max_tokens=8000,
                    )
                except Exception:
                    self._set_status(name, "shutdown")
                    return
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    break
                results = []
                idle_requested = False
                for block in response.content:
                    if block.type == "tool_use":
                        if block.name == "idle":
                            idle_requested = True
                            output = "Entering idle phase."
                        elif block.name == "claim_task":
                            output = self.task_mgr.claim(
                                block.input["task_id"], name
                            )
                        elif block.name == "send_message":
                            output = self.bus.send(
                                name, block.input["to"], block.input["content"]
                            )
                        else:
                            dispatch = {
                                "bash": lambda **kw: run_bash(kw["command"]),
                                "read_file": lambda **kw: run_read(kw["path"]),
                                "write_file": lambda **kw: run_write(
                                    kw["path"], kw["content"]
                                ),
                                "edit_file": lambda **kw: run_edit(
                                    kw["path"], kw["old_text"], kw["new_text"]
                                ),
                            }
                            output = dispatch.get(
                                block.name, lambda **kw: "Unknown"
                            )(**block.input)
                        print(f"  [{name}] {block.name}: {str(output)[:120]}")
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(output),
                            }
                        )
                messages.append({"role": "user", "content": results})
                if idle_requested:
                    break
            # -- IDLE PHASE: poll for messages and unclaimed tasks --
            self._set_status(name, "idle")
            resume = False
            for _ in range(IDLE_TIMEOUT // max(POLL_INTERVAL, 1)):
                time.sleep(POLL_INTERVAL)
                inbox = self.bus.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break
                unclaimed = []
                for f in sorted(TASKS_DIR.glob("task_*.json")):
                    t = json.loads(f.read_text())
                    if (
                        t.get("status") == "pending"
                        and not t.get("owner")
                        and not t.get("blockedBy")
                    ):
                        unclaimed.append(t)
                if unclaimed:
                    task = unclaimed[0]
                    self.task_mgr.claim(task["id"], name)
                    # Identity re-injection for compressed contexts
                    if len(messages) <= 3:
                        messages.insert(
                            0,
                            {
                                "role": "user",
                                "content": (
                                    f"<identity>You are '{name}', role: {role}, "
                                    f"team: {team_name}.</identity>"
                                ),
                            },
                        )
                        messages.insert(
                            1,
                            {
                                "role": "assistant",
                                "content": f"I am {name}. Continuing.",
                            },
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
                                f"{task.get('description', '')}</auto-claimed>"
                            ),
                        }
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"Claimed task #{task['id']}. Working on it.",
                        }
                    )
                    resume = True
                    break
            if not resume:
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]
