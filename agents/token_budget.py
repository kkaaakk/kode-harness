from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


SUMMARY_TAG = "conversation_summary"


@dataclass(frozen=True)
class TokenBudgetConfig:
    max_context_tokens: int = 100000
    compress_threshold_ratio: float = 0.85
    recent_steps_keep_count: int = 3
    summary_max_tokens: int = 2000
    model_name: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        max_context_tokens: int = 100000,
        compress_threshold_ratio: float = 0.85,
        recent_steps_keep_count: int = 3,
        summary_max_tokens: int = 2000,
        model_name: str | None = None,
    ) -> "TokenBudgetConfig":
        return cls(
            max_context_tokens=_env_int(
                "TOKEN_BUDGET_MAX_CONTEXT_TOKENS", max_context_tokens
            ),
            compress_threshold_ratio=_env_float(
                "TOKEN_BUDGET_COMPRESS_THRESHOLD_RATIO", compress_threshold_ratio
            ),
            recent_steps_keep_count=_env_int(
                "TOKEN_BUDGET_RECENT_STEPS_KEEP_COUNT", recent_steps_keep_count
            ),
            summary_max_tokens=_env_int(
                "TOKEN_BUDGET_SUMMARY_MAX_TOKENS", summary_max_tokens
            ),
            model_name=os.getenv("TOKEN_BUDGET_MODEL_NAME", model_name or "") or None,
        )

    @property
    def threshold_tokens(self) -> int:
        return int(self.max_context_tokens * self.compress_threshold_ratio)


@dataclass(frozen=True)
class BudgetCheckReport:
    triggered: bool
    before_tokens: int
    after_tokens: int
    threshold_tokens: int
    transcript_path: Path | None = None


@dataclass(frozen=True)
class CompactionReport:
    before_tokens: int
    after_tokens: int
    transcript_path: Path
    older_message_count: int
    recent_message_count: int
    protected_message_count: int


@dataclass
class _PendingToolResult:
    result: dict[str, Any]
    tool_name: str


@dataclass
class MicroCompactState:
    last_message_index: int = 0
    last_scanned_message_id: int | None = None
    pending_tool_results: list[_PendingToolResult] = field(default_factory=list)
    tool_names_by_id: dict[str, str] = field(default_factory=dict)

    def reset(self) -> None:
        self.last_message_index = 0
        self.last_scanned_message_id = None
        self.pending_tool_results.clear()
        self.tool_names_by_id.clear()


SummarizeFn = Callable[..., str]
MicroCompactReplacementFn = Callable[[str], str]


def estimate_tokens(payload: Any, model_name: str | None = None) -> int:
    text = payload if isinstance(payload, str) else _json_dumps(payload)
    encoder = _get_tiktoken_encoder(model_name)
    if encoder is not None:
        return len(encoder.encode(text))
    return max(1, len(text) // 4)


def micro_compact_tool_results(
    messages: list[dict[str, Any]],
    *,
    state: MicroCompactState,
    keep_recent: int,
    preserve_result_tools: set[str] | None = None,
    replacement_for_tool: MicroCompactReplacementFn | None = None,
    min_content_chars: int = 100,
) -> list[dict[str, Any]]:
    if _micro_compact_state_is_stale(messages, state):
        state.reset()

    preserve_result_tools = preserve_result_tools or set()
    replacement_for_tool = replacement_for_tool or _default_tool_result_replacement
    start_index = state.last_message_index

    for message in messages[start_index:]:
        content = message.get("content")
        if message.get("role") == "assistant" and isinstance(content, list):
            for block in content:
                if _block_type(block) == "tool_use":
                    tool_id = _block_value(block, "id")
                    if tool_id is not None:
                        tool_name = _block_value(block, "name") or "unknown"
                        state.tool_names_by_id[str(tool_id)] = str(tool_name)
        elif message.get("role") == "user" and isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_id = str(part.get("tool_use_id", ""))
                    tool_name = state.tool_names_by_id.get(tool_id, "unknown")
                    state.pending_tool_results.append(
                        _PendingToolResult(result=part, tool_name=tool_name)
                    )

    state.last_message_index = len(messages)
    state.last_scanned_message_id = id(messages[-1]) if messages else None

    keep_recent = max(keep_recent, 0)
    while len(state.pending_tool_results) > keep_recent:
        pending = state.pending_tool_results.pop(0)
        if pending.tool_name in preserve_result_tools:
            continue
        content = pending.result.get("content")
        if isinstance(content, str) and len(content) > min_content_chars:
            pending.result["content"] = replacement_for_tool(pending.tool_name)

    return messages


def prepare_messages_for_model(
    messages: list[dict[str, Any]],
    *,
    config: TokenBudgetConfig,
    summarize: SummarizeFn,
    transcript_dir: Path,
    logger: Callable[[str], None] = print,
) -> BudgetCheckReport:
    before_tokens = estimate_tokens(messages, config.model_name)
    threshold_tokens = config.threshold_tokens
    if before_tokens < threshold_tokens:
        return BudgetCheckReport(
            triggered=False,
            before_tokens=before_tokens,
            after_tokens=before_tokens,
            threshold_tokens=threshold_tokens,
        )

    logger(
        "[token-budget] compression triggered: "
        f"tokens={before_tokens}, threshold={threshold_tokens}, "
        f"max={config.max_context_tokens}"
    )
    report = compact_messages(
        messages,
        config=config,
        summarize=summarize,
        transcript_dir=transcript_dir,
    )
    logger(
        "[token-budget] compression complete: "
        f"tokens {report.before_tokens} -> {report.after_tokens}; "
        f"older_messages={report.older_message_count}; "
        f"recent_messages={report.recent_message_count}; "
        f"protected_memory={report.protected_message_count}; "
        f"transcript={report.transcript_path}"
    )
    return BudgetCheckReport(
        triggered=True,
        before_tokens=report.before_tokens,
        after_tokens=report.after_tokens,
        threshold_tokens=threshold_tokens,
        transcript_path=report.transcript_path,
    )


def compact_messages(
    messages: list[dict[str, Any]],
    *,
    config: TokenBudgetConfig,
    summarize: SummarizeFn,
    transcript_dir: Path,
) -> CompactionReport:
    before_tokens = estimate_tokens(messages, config.model_name)
    transcript_path = save_transcript(messages, transcript_dir)
    protected, previous_summaries, candidates = _extract_memory_layers(messages)
    older, recent = _split_recent_steps(
        candidates, keep_steps=config.recent_steps_keep_count
    )
    previous_summary = "\n\n".join(previous_summaries).strip()
    history_text = _json_dumps(older)
    summary = summarize(
        previous_summary=previous_summary,
        history_text=history_text,
        summary_max_tokens=config.summary_max_tokens,
    ).strip()
    if not summary:
        summary = previous_summary or "No summary generated."

    messages[:] = protected + [make_summary_message(summary, transcript_path)] + recent
    after_tokens = estimate_tokens(messages, config.model_name)
    return CompactionReport(
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        transcript_path=transcript_path,
        older_message_count=len(older),
        recent_message_count=len(recent),
        protected_message_count=len(protected),
    )


def save_transcript(messages: list[dict[str, Any]], transcript_dir: Path) -> Path:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f"transcript_{int(time.time() * 1000)}.jsonl"
    with open(transcript_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(_json_dumps(message) + "\n")
    return transcript_path


def make_summary_message(summary: str, transcript_path: Path | str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"[Conversation compressed. Transcript: {transcript_path}]\n"
            f"<{SUMMARY_TAG}>\n"
            f"{summary}\n"
            f"</{SUMMARY_TAG}>"
        ),
    }


def summarize_with_anthropic(
    client: Any,
    *,
    model: str,
    previous_summary: str,
    history_text: str,
    summary_max_tokens: int,
) -> str:
    prompt = (
        "Update the rolling conversation summary for continuity.\n\n"
        "Rules:\n"
        "- Preserve important goals, decisions, open tasks, file paths, and current state.\n"
        "- Merge the previous summary with the older history; do not restart from scratch.\n"
        "- Protected user preferences and long-term memory are preserved separately, "
        "so do not rewrite or override them here.\n"
        "- Keep the result concise and useful for the next model call.\n\n"
        f"<previous_summary>\n{previous_summary or '(none)'}\n</previous_summary>\n\n"
        f"<older_history>\n{history_text}\n</older_history>"
    )
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=summary_max_tokens,
    )
    return _response_text(response)


def _extract_memory_layers(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    protected: list[dict[str, Any]] = []
    previous_summaries: list[str] = []
    candidates: list[dict[str, Any]] = []
    seen_protected = set()
    for message in messages:
        if is_protected_memory_message(message):
            marker = _json_dumps(message)
            if marker not in seen_protected:
                protected.append(message)
                seen_protected.add(marker)
        elif is_summary_message(message):
            summary = extract_summary_text(message)
            if summary:
                previous_summaries.append(summary)
        else:
            candidates.append(message)
    return protected, previous_summaries, candidates


def _split_recent_steps(
    messages: list[dict[str, Any]], *, keep_steps: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if keep_steps <= 0:
        return messages[:], []
    starts = [
        index for index, message in enumerate(messages) if _is_step_start(message)
    ]
    if len(starts) <= keep_steps:
        return [], messages[:]
    split_at = starts[-keep_steps]
    return messages[:split_at], messages[split_at:]


def _is_step_start(message: dict[str, Any]) -> bool:
    return message.get("role") == "user" and not _is_tool_result_message(message)


def _is_tool_result_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(_block_type(part) == "tool_result" for part in content)


def is_summary_message(message: dict[str, Any]) -> bool:
    metadata = message.get("metadata")
    if isinstance(metadata, dict) and metadata.get("kind") == SUMMARY_TAG:
        return True
    text = _content_text(message.get("content"))
    return (
        f"<{SUMMARY_TAG}" in text
        or "[Compressed." in text
        or "[Conversation compressed." in text
    )


def extract_summary_text(message: dict[str, Any]) -> str:
    text = _content_text(message.get("content")).strip()
    match = re.search(
        rf"<{SUMMARY_TAG}[^>]*>(.*?)</{SUMMARY_TAG}>",
        text,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return text


def is_protected_memory_message(message: dict[str, Any]) -> bool:
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        memory_type = str(metadata.get("memory_type", "")).lower()
        if memory_type in {"preference", "user_preference", "long_term", "protected"}:
            return True
    return False


def _block_type(block: Any) -> str | None:
    if isinstance(block, dict):
        block_type = block.get("type")
    else:
        block_type = getattr(block, "type", None)
    return str(block_type) if block_type is not None else None


def _block_value(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _default_tool_result_replacement(tool_name: str) -> str:
    return "[cleared]"


def _micro_compact_state_is_stale(
    messages: list[dict[str, Any]], state: MicroCompactState
) -> bool:
    if state.last_message_index == 0:
        return False
    if state.last_message_index > len(messages):
        return True
    if not messages:
        return True
    return id(messages[state.last_message_index - 1]) != state.last_scanned_message_id


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return _json_dumps(content)


def _response_text(response: Any) -> str:
    chunks = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text is not None:
            chunks.append(text)
        elif isinstance(block, dict) and "text" in block:
            chunks.append(str(block["text"]))
    return "".join(chunks)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        data = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        if data:
            return data
    return str(value)


def _get_tiktoken_encoder(model_name: str | None) -> Any:
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None
    try:
        if model_name:
            return tiktoken.encoding_for_model(model_name)
    except Exception:
        pass
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default
