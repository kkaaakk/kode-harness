from __future__ import annotations

import json
from pathlib import Path

from agents.trace_context import (
    clear_trace,
    get_current_span_id,
    get_trace_id,
    set_current_span_id,
    start_trace,
)
from agents.tracer import record_event, trace_span


def read_trace_events(trace_dir: Path) -> list[dict]:
    events = []
    for path in sorted(trace_dir.glob("trace_*.jsonl")):
        events.extend(json.loads(line) for line in path.read_text().splitlines())
    return events


def test_trace_context_stores_current_trace_and_span() -> None:
    clear_trace()

    trace_id = start_trace()

    assert trace_id
    assert get_trace_id() == trace_id
    assert get_current_span_id() == "root"

    set_current_span_id("llm_1")
    assert get_current_span_id() == "llm_1"

    clear_trace()
    assert get_trace_id() is None
    assert get_current_span_id() is None


def test_trace_span_records_duration_and_restores_parent(tmp_path: Path) -> None:
    clear_trace()
    trace_id = start_trace()

    with trace_span(
        "llm_call",
        trace_dir=tmp_path,
        input_summary={"messages": 2},
    ) as span:
        assert get_current_span_id() == span.span_id
        span.set_output({"stop_reason": "end_turn"})

    assert get_current_span_id() == "root"
    events = read_trace_events(tmp_path)

    assert len(events) == 1
    event = events[0]
    assert event["trace_id"] == trace_id
    assert event["span_id"] == span.span_id
    assert event["parent_span_id"] == "root"
    assert event["event"] == "llm_call"
    assert event["status"] == "success"
    assert event["start_time"]
    assert event["duration_ms"] >= 0
    assert "messages" in event["input_summary"]
    assert "end_turn" in event["output_summary"]

    clear_trace()


def test_record_event_noops_without_trace(tmp_path: Path) -> None:
    clear_trace()

    record_event("orphan", trace_dir=tmp_path)

    assert read_trace_events(tmp_path) == []
