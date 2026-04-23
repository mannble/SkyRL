from __future__ import annotations

import json
import re
from typing import Any

from skyrl_agent.meta_toolkit.observability.trace_schema import TraceEvent, TraceRecord


_EXTRA_JSON_TEXT_RE = re.compile(
    r"-\s*Extra text detected (?:before|after) JSON object",
    flags=re.IGNORECASE,
)


def _iter_balanced_json_objects(text: str):
    """Yield balanced JSON object candidates while respecting quoted strings."""
    start: int | None = None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if in_string:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue
        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : i + 1]
                start = None


def _extract_first_valid_json_object(text: str) -> str | None:
    for candidate in _iter_balanced_json_objects(text):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return candidate
        except json.JSONDecodeError:
            continue
    return None


def _strip_parser_warning_noise(text: str) -> str:
    cleaned = text
    cleaned = re.sub(
        r"Previous response had warnings:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"WARNINGS:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = _EXTRA_JSON_TEXT_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() or text


def _sanitize_meta_event_content(role: str, content: str) -> str:
    """Reduce trace noise before feeding events to meta diagnosis/planning."""
    cleaned = _strip_parser_warning_noise(content)
    if role == "assistant":
        json_object = _extract_first_valid_json_object(cleaned)
        if json_object:
            return json_object
    return cleaned


def _infer_failure_tags(
    stop_reason: str,
    successful: bool,
    results: Any,
    *,
    reward: float,
    messages: list[dict[str, Any]],
) -> list[str]:
    tags: list[str] = []
    if successful:
        return tags

    exc_type = None
    exc_message = ""
    if results is not None and getattr(results, "exception_info", None) is not None:
        exc_type = getattr(results.exception_info, "exception_type", None)
        exc_message = str(getattr(results.exception_info, "exception_message", "") or "")

    message_blob = "\n".join(
        str(m.get("content", "")) for m in messages if isinstance(m, dict)
    )
    has_parse_errors = (
        "Invalid JSON" in message_blob
        or "Missing required fields" in message_blob
        or "parsing errors" in message_blob
    )

    if stop_reason == "agent_timeout":
        tags.extend(["termination_failure", "sync_bottleneck"])
    elif stop_reason == "context_length":
        tags.append("context_overload")
    elif stop_reason == "complete":
        # The run ended cleanly but still got reward 0 => usually verification/task-quality gap.
        if reward <= 0:
            tags.append("verification_failure")
            if has_parse_errors:
                tags.append("command_parsing_failure")
        else:
            tags.append("task_failure")
    else:
        # Runtime/setup/tool failures.
        tags.append("recovery_failure")
        if has_parse_errors:
            tags.append("command_parsing_failure")
        if (
            stop_reason == "error"
            and exc_type == "AttributeError"
            and "NoneType" in exc_message
            and "strip" in exc_message
        ):
            tags.append("initialization_failure")

    if exc_type == "ContextLengthExceededError" and "context_overload" not in tags:
        tags.append("context_overload")
    if exc_type == "AgentTimeoutError" and "sync_bottleneck" not in tags:
        tags.append("sync_bottleneck")
    if exc_type and "Timeout" in exc_type and "sync_bottleneck" not in tags:
        tags.append("sync_bottleneck")
    # Keep insertion order while deduplicating.
    return list(dict.fromkeys(tags))


def build_trace_from_harbor_trial(
    *,
    trajectory_id: Any,
    agent_version: str,
    prompt: Any,
    results: Any,
    reward: float,
    stop_reason: str,
    num_turns: int,
    summarization_count: int,
    successful: bool,
) -> TraceRecord:
    """Build a minimal TraceRecord from Harbor trial outputs."""

    task_id = f"{trajectory_id.instance_id}-traj{trajectory_id.repetition_id}"
    messages = []
    if results is not None and getattr(results, "agent_result", None) is not None:
        messages = results.agent_result.metadata.get("all_messages", []) or []

    events: list[TraceEvent] = []
    for turn_id, message in enumerate(messages):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if isinstance(content, str):
            content = _sanitize_meta_event_content(role, content)
        action_type = "thought"
        if role == "tool":
            action_type = "tool_result"
        events.append(
            TraceEvent(
                turn_id=turn_id,
                role=role,
                action_type=action_type,
                content=content if isinstance(content, str) else str(content),
            )
        )

    tool_calls = sum(1 for message in messages if message.get("role") == "assistant")
    tool_failures = 0 if successful else int(stop_reason in {"error", "agent_timeout"})
    sync_bottleneck_score = 1.0 if stop_reason == "agent_timeout" else 0.0
    if results is not None and getattr(results, "exception_info", None) is not None:
        exc_type = getattr(results.exception_info, "exception_type", None)
        if exc_type == "AgentTimeoutError":
            sync_bottleneck_score = 1.0

    return TraceRecord(
        task_id=task_id,
        agent_version=agent_version,
        policy_version="unknown",
        success=successful,
        final_reward=float(reward),
        turn_count=num_turns,
        finish_reason=stop_reason,
        sync_bottleneck_score=sync_bottleneck_score,
        failure_tags=_infer_failure_tags(
            stop_reason,
            successful,
            results,
            reward=reward,
            messages=messages,
        ),
        tool_calls=tool_calls,
        tool_failures=tool_failures,
        verification_attempted=False,
        retry_attempted=False,
        events=events,
        metadata={
            "prompt": prompt,
            "summarization_count": summarization_count,
        },
    )
