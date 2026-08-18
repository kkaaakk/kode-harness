from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from agents.trace_context import (
        get_current_span_id,
        get_trace_id,
        set_current_span_id,
    )
except ImportError:
    from trace_context import (
        get_current_span_id,
        get_trace_id,
        set_current_span_id,
    )


DEFAULT_TRACE_DIR = Path(".tmp") / "runtime" / "team" / "traces"
SUMMARY_LIMIT = 1200
_write_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def summarize_value(value: Any, limit: int = SUMMARY_LIMIT) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=True, default=str)
        except TypeError:
            text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... ({len(text) - limit} more chars)"


def new_span_id(event: str) -> str:
    prefix = "".join(ch if ch.isalnum() else "_" for ch in event.lower()).strip("_")
    prefix = prefix or "span"
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def trace_file_path(trace_dir: Path | None = None) -> Path:
    root = trace_dir or DEFAULT_TRACE_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root / f"trace_{datetime.now(timezone.utc):%Y%m%d}.jsonl"


def record_event(
    event: str,
    *,
    status: str = "success",
    span_id: str | None = None,
    parent_span_id: str | None = None,
    start_time: str | None = None,
    duration_ms: float | int | None = None,
    input_summary: Any = None,
    output_summary: Any = None,
    error: str | None = None,
    trace_dir: Path | None = None,
    **fields: Any,
) -> None:
    trace_id = get_trace_id()
    if trace_id is None:
        return

    row = {
        "trace_id": trace_id,
        "span_id": span_id or new_span_id(event),
        "parent_span_id": parent_span_id,
        "event": event,
        "status": status,
        "time": utc_now(),
        "start_time": start_time,
        "duration_ms": duration_ms,
        "input_summary": summarize_value(input_summary),
        "output_summary": summarize_value(output_summary),
        "error": error,
    }
    row.update(fields)

    path = trace_file_path(trace_dir)
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")


class TraceSpan:
    def __init__(
        self,
        event: str,
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        input_summary: Any = None,
        trace_dir: Path | None = None,
        inherit_parent: bool = True,
        **fields: Any,
    ):
        self.event = event
        self.span_id = span_id or new_span_id(event)
        self.parent_span_id = parent_span_id
        self.input_summary = input_summary
        self.inherit_parent = inherit_parent
        self.output_summary: Any = None
        self.status = "success"
        self.error: str | None = None
        self.trace_dir = trace_dir
        self.fields = fields
        self.start_time = ""
        self._started = 0.0
        self._previous_span_id: str | None = None

    def __enter__(self) -> "TraceSpan":
        self._previous_span_id = get_current_span_id()
        if self.parent_span_id is None and self.inherit_parent:
            self.parent_span_id = self._previous_span_id
        self.start_time = utc_now()
        self._started = time.perf_counter()
        set_current_span_id(self.span_id)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        duration_ms = round((time.perf_counter() - self._started) * 1000, 2)
        record_event(
            self.event,
            status="error" if exc else self.status,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            start_time=self.start_time,
            duration_ms=duration_ms,
            input_summary=self.input_summary,
            output_summary=self.output_summary,
            error=str(exc) if exc else self.error,
            trace_dir=self.trace_dir,
            **self.fields,
        )
        if self._previous_span_id is not None:
            set_current_span_id(self._previous_span_id)
        return False

    def set_output(self, value: Any) -> None:
        self.output_summary = value

    def set_status(self, status: str) -> None:
        self.status = status

    def set_error(self, error: Exception | str) -> None:
        self.status = "error"
        self.error = str(error)


def trace_span(
    event: str,
    *,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    input_summary: Any = None,
    trace_dir: Path | None = None,
    inherit_parent: bool = True,
    **fields: Any,
) -> TraceSpan:
    return TraceSpan(
        event,
        span_id=span_id,
        parent_span_id=parent_span_id,
        input_summary=input_summary,
        trace_dir=trace_dir,
        inherit_parent=inherit_parent,
        **fields,
    )
