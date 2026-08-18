from __future__ import annotations

import uuid
from contextvars import ContextVar


_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_current_span_id: ContextVar[str | None] = ContextVar("current_span_id", default=None)


def start_trace() -> str:
    trace_id = uuid.uuid4().hex[:16]
    _trace_id.set(trace_id)
    _current_span_id.set("root")
    return trace_id


def get_trace_id() -> str | None:
    return _trace_id.get()


def get_current_span_id() -> str | None:
    return _current_span_id.get()


def set_current_span_id(span_id: str) -> None:
    _current_span_id.set(span_id)


def clear_trace() -> None:
    _trace_id.set(None)
    _current_span_id.set(None)


reset_trace = clear_trace
